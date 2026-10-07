import json
import logging
import os
import sqlite3
import threading
import time
import uuid
import zipfile
from pathlib import Path

from PIL import Image, ImageOps
from .compressor import encode_image
from .image_utils import inspect_image, reserve_output, safe_name
from .upscaler import Upscaler, Cancelled

TERMINAL = {'Completed', 'Failed', 'Cancelled'}
DEFAULTS = dict(scale=2, quality='high', dpi=False, compress=False, compression='balanced',
                target_mb=None, format='original', base_name='', output_dir='')


def validate_options(raw):
    options = {**DEFAULTS, **{key: value for key, value in raw.items() if key in DEFAULTS}}
    if options['scale'] not in (2, 3, 4) or options['quality'] not in ('fast', 'balanced', 'high'):
        raise ValueError('Invalid scale or quality')
    if options['format'] not in ('original', 'JPEG', 'PNG', 'WEBP'):
        raise ValueError('Invalid output format')
    if options['compression'] not in ('light', 'balanced', 'maximum'):
        raise ValueError('Invalid compression level')
    if not isinstance(options['dpi'], bool) or not isinstance(options['compress'], bool):
        raise ValueError('Invalid toggle value')
    if options['target_mb'] is not None:
        options['target_mb'] = float(options['target_mb'])
        if not 0.01 <= options['target_mb'] <= 1000:
            raise ValueError('Target must be between 0.01 and 1000 MB')
    for key in ('base_name', 'output_dir'):
        if not isinstance(options[key], str):
            raise ValueError('Invalid filename or folder')
    return options


class QueueManager:
    def __init__(self, root, engine=None):
        self.root = Path(root).resolve()
        for folder in ('temp/uploads', 'temp/thumbnails', 'temp/exports', 'output/completed', 'output/failed', 'logs', 'models'):
            (self.root / folder).mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.root / 'temp/session.sqlite3', check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, data TEXT NOT NULL)')
        self.db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value INTEGER NOT NULL)')
        self.db.commit()
        self.lock = threading.RLock()
        self.jobs = {key: json.loads(data) for key, data in self.db.execute('SELECT id, data FROM jobs ORDER BY rowid')}
        for job in self.jobs.values():
            if job['status'] not in TERMINAL and job['status'] != 'Waiting':
                job.update(status='Waiting', progress=0, error='Recovered after restart')
                self._save(job)
            if job['status'] == 'Completed' and not Path(job.get('output', '')).is_file():
                job.update(status='Failed', error='Output file is missing; retry to regenerate.')
                self._save(job)
        counter = self.db.execute("SELECT value FROM metadata WHERE key='sequence'").fetchone()
        self.sequence = max(counter[0] if counter else 0, max((j.get('sequence', 0) for j in self.jobs.values()), default=0))
        self.engine = engine or Upscaler(self.root / 'models')
        self.running = False
        self.paused = False
        self.active = None
        self.cancelled = set()
        self.removing = set()
        self.shutdown = threading.Event()
        self.zip_state = dict(status='Idle', progress=0, error='', id=None)
        self.thread = threading.Thread(target=self._worker, daemon=True, name='image-worker')
        self.thread.start()

    def _save(self, job):
        self.db.execute('INSERT OR REPLACE INTO jobs VALUES (?, ?)', (job['id'], json.dumps(job)))
        self.db.commit()

    def _update(self, job, **changes):
        with self.lock:
            job.update(changes)
            self._save(job)

    def add(self, upload, options):
        options = validate_options(options)
        identifier = uuid.uuid4().hex
        name = safe_name(Path(upload.filename.replace('\\', '/')).name)
        source = self.root / 'temp/uploads' / identifier
        thumb = self.root / 'temp/thumbnails' / f'{identifier}.png'
        upload.save(source)
        with self.lock:
            self.sequence += 1
            self.db.execute("INSERT OR REPLACE INTO metadata VALUES ('sequence', ?)", (self.sequence,))
            job = dict(id=identifier, filename=name, source=str(source), thumbnail=str(thumb),
                       sequence=self.sequence, options=options, width=0, height=0, original_format='', alpha=False,
                       original_size=source.stat().st_size, output_size=0, output='', status='Waiting',
                       progress=0, error='', warning='')
            try:
                job['width'], job['height'], job['original_format'], job['alpha'] = inspect_image(source, thumb)
                if job['width'] * job['height'] * 16 > 100_000_000:
                    raise ValueError('Image exceeds the safe native 4x limit (100 megapixels output).')
            except Exception as exc:
                job.update(status='Failed', error=f'Unsupported or damaged image: {exc}')
            self.jobs[identifier] = job
            self._save(job)
        return job['id']

    def snapshot(self):
        with self.lock:
            jobs = []
            for job in self.jobs.values():
                item = {k: v for k, v in job.items() if k not in ('source', 'thumbnail', 'output')}
                item['output_name'] = Path(job['output']).name if job['output'] else ''
                item['output_width'] = job['width'] * job['options']['scale']
                item['output_height'] = job['height'] * job['options']['scale']
                jobs.append(item)
            return dict(jobs=jobs, running=self.running, paused=self.paused, active=self.active,
                        completed=sum(j['status'] == 'Completed' for j in jobs), total=len(jobs),
                        device=self.engine.device, model_progress=self.engine.download_progress, zip=dict(self.zip_state))

    def control(self, action, ids=None, options=None):
        with self.lock:
            if action == 'start':
                self.running, self.paused = True, False
            elif action == 'pause':
                self.paused = True
            elif action == 'resume':
                self.running, self.paused = True, False
            elif action in ('cancel', 'cancel_current', 'remove', 'clear', 'clear_completed', 'retry', 'retry_failed', 'apply'):
                if action == 'cancel_current':
                    selected = [self.active] if self.active else []
                elif action == 'clear':
                    selected = list(self.jobs)
                elif action == 'clear_completed':
                    selected = [k for k, j in self.jobs.items() if j['status'] == 'Completed']
                elif action == 'retry_failed':
                    selected = [k for k, j in self.jobs.items() if j['status'] == 'Failed']
                else:
                    selected = ids or []
                new_options = validate_options(options or {}) if action == 'apply' else None
                for key in selected:
                    job = self.jobs.get(key)
                    if not job:
                        continue
                    if action in ('remove', 'clear', 'clear_completed'):
                        if key == self.active:
                            self.cancelled.add(key)
                            self.removing.add(key)
                        else:
                            self._remove(key)
                    elif action in ('cancel', 'cancel_current'):
                        if key == self.active:
                            self.cancelled.add(key)
                        elif job['status'] == 'Waiting':
                            self._update(job, status='Cancelled')
                    elif action in ('retry', 'retry_failed') and job['status'] in ('Failed', 'Cancelled') and key != self.active:
                        self.cancelled.discard(key)
                        self._update(job, status='Waiting', progress=0, error='', warning='')
                    elif action == 'apply' and job['status'] == 'Waiting':
                        self._update(job, options=new_options)
            else:
                raise ValueError('Unknown queue action')

    def _remove(self, key):
        job = self.jobs.pop(key)
        # Removing a queue entry never deletes a completed production asset.
        Path(job['source']).unlink(missing_ok=True)
        Path(job['thumbnail']).unlink(missing_ok=True)
        self.db.execute('DELETE FROM jobs WHERE id=?', (key,))
        self.db.commit()
        self.cancelled.discard(key)
        self.removing.discard(key)

    def _checkpoint(self, key):
        while True:
            with self.lock:
                if self.shutdown.is_set() or key in self.cancelled:
                    raise Cancelled()
                paused = self.paused
            if not paused:
                return
            self.shutdown.wait(.15)

    def _worker(self):
        while not self.shutdown.is_set():
            job = None
            with self.lock:
                if self.running and not self.paused:
                    job = next((j for j in self.jobs.values() if j['status'] == 'Waiting'), None)
                    if job:
                        self.active = job['id']
                        self._update(job, status='Processing', progress=1, error='')
                    else:
                        self.running = False
            if job is None:
                self.shutdown.wait(.15)
                continue
            key = job['id']
            output = None
            try:
                checkpoint = lambda: self._checkpoint(key)
                checkpoint()
                opts = job['options']
                with Image.open(job['source']) as original:
                    image = ImageOps.exif_transpose(original).copy()
                if job['alpha']:
                    image = image.convert('RGBA')
                self._update(job, status='Upscaling')
                result = self.engine.upscale(image, opts['scale'], opts['quality'],
                                             lambda p: self._update(job, progress=round(p)), checkpoint)
                checkpoint()
                self._update(job, status='Compressing' if opts['compress'] else 'Processing', progress=92)
                fmt = job['original_format'] if opts['format'] == 'original' else opts['format']
                data, warning = encode_image(result, fmt, opts['dpi'], opts['compress'], opts['compression'], opts['target_mb'], checkpoint)
                result.close()
                image.close()
                checkpoint()
                folder = Path(opts['output_dir']).expanduser() if opts['output_dir'] else self.root / 'output'
                if not folder.is_absolute():
                    folder = self.root / folder
                stem = f"{safe_name(opts['base_name'])}-{job['sequence']:03d}" if opts['base_name'].strip() else safe_name(Path(job['filename']).stem)
                suffix = {'JPEG': 'jpg', 'PNG': 'png', 'WEBP': 'webp'}[fmt]
                output = reserve_output(folder / 'completed', stem, suffix)
                temporary = output.with_suffix(output.suffix + '.part')
                try:
                    with temporary.open('wb') as stream:
                        stream.write(data)
                        stream.flush()
                        os.fsync(stream.fileno())
                    checkpoint()
                    os.replace(temporary, output)
                finally:
                    temporary.unlink(missing_ok=True)
                self._update(job, status='Completed', progress=100, output=str(output), output_size=len(data), warning=warning)
            except Cancelled:
                if output:
                    output.unlink(missing_ok=True)
                self._update(job, status='Cancelled', progress=0)
            except Exception as exc:
                if output:
                    output.unlink(missing_ok=True)
                logging.exception('Job failed: %s', job['filename'])
                self._update(job, status='Failed', error=str(exc)[:250])
                try:
                    (self.root / 'output/failed' / f'{key}.json').write_text(json.dumps({'filename': job['filename'], 'error': str(exc)}, indent=2), encoding='utf-8')
                except OSError:
                    logging.exception('Could not write failure report')
            finally:
                with self.lock:
                    self.active = None
                    self.cancelled.discard(key)
                    if key in self.removing:
                        self._remove(key)

    def job_path(self, key, kind='output'):
        with self.lock:
            job = self.jobs.get(key)
            if not job or (kind == 'output' and job['status'] != 'Completed'):
                raise FileNotFoundError('Output is not ready')
            path = Path(job[kind])
            if not path.is_file():
                raise FileNotFoundError('File is missing')
            return path

    def delete_output(self, key):
        with self.lock:
            self.job_path(key).unlink()
            self._update(self.jobs[key], status='Cancelled', output='', output_size=0, progress=0)

    def start_zip(self):
        with self.lock:
            if self.zip_state['status'] == 'Processing':
                return
            paths = [Path(j['output']) for j in self.jobs.values() if j['status'] == 'Completed' and Path(j['output']).is_file()]
            if not paths:
                raise ValueError('No completed images to export')
            identifier = uuid.uuid4().hex
            self.zip_state = dict(status='Processing', progress=0, error='', id=identifier)
            threading.Thread(target=self._zip, args=(paths, identifier), daemon=True).start()

    def _zip(self, paths, identifier):
        target = self.root / 'temp/exports' / f'{identifier}.zip'
        partial = target.with_suffix('.part')
        try:
            names = set()
            with zipfile.ZipFile(partial, 'w', compression=zipfile.ZIP_STORED, allowZip64=True) as archive:
                for index, path in enumerate(paths):
                    name = path.name
                    duplicate = 1
                    while name in names:
                        name = f'{path.stem}-{duplicate:03d}{path.suffix}'
                        duplicate += 1
                    names.add(name)
                    archive.write(path, name)
                    with self.lock:
                        self.zip_state['progress'] = round((index + 1) * 100 / len(paths))
            os.replace(partial, target)
            with self.lock:
                self.zip_state['status'] = 'Completed'
        except Exception as exc:
            logging.exception('ZIP export failed')
            with self.lock:
                self.zip_state.update(status='Failed', error=str(exc)[:200])
        finally:
            partial.unlink(missing_ok=True)

    def close(self):
        self.shutdown.set()
        self.thread.join(timeout=30)
        if not self.thread.is_alive():
            self.db.close()
