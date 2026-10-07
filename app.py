"""Local-only application. Run: python app.py (Windows: run.bat)."""
import argparse
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import secrets
import subprocess
import sys
import threading
import webbrowser

try:
    from flask import Flask, jsonify, render_template, request, send_file
    from waitress import serve
    from filelock import FileLock, Timeout
    import torch
    import cv2
    from core.queue_manager import QueueManager
except (ImportError, OSError) as exc:
    print(f'Missing dependency: {exc}. Run install.bat, or pip install -r requirements.txt.')
    raise SystemExit(1)

ROOT = Path(__file__).resolve().parent


def create_app(root=ROOT, engine=None):
    app = Flask(__name__, template_folder=str(ROOT / 'templates'), static_folder=str(ROOT / 'static'))
    app.config.update(MAX_CONTENT_LENGTH=1024 * 1024 * 1024, MAX_FORM_PARTS=10000, TRUSTED_HOSTS=['127.0.0.1', 'localhost', '[::1]'])
    root = Path(root)
    (root / 'logs').mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(root / 'logs/app.log', maxBytes=5_000_000, backupCount=3, encoding='utf-8')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logging.getLogger().addHandler(handler)
    logging.getLogger().setLevel(logging.INFO)
    manager = QueueManager(root, engine)
    token = secrets.token_urlsafe(32)
    app.extensions['queue'] = manager
    app.extensions['csrf_token'] = token

    @app.before_request
    def protect_local_actions():
        if request.method not in ('GET', 'HEAD', 'OPTIONS'):
            if not secrets.compare_digest(request.headers.get('X-App-Token', ''), token):
                return jsonify(error='Invalid local session. Reload the application.'), 403

    @app.after_request
    def headers(response):
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['X-Frame-Options'] = 'DENY'
        response.headers['Referrer-Policy'] = 'same-origin'
        response.headers['Content-Security-Policy'] = "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        if request.path.startswith('/api'):
            response.headers['Cache-Control'] = 'no-store'
        return response

    @app.errorhandler(Exception)
    def error(exc):
        from werkzeug.exceptions import HTTPException
        code = exc.code if isinstance(exc, HTTPException) else 400 if isinstance(exc, (ValueError, FileNotFoundError)) else 500
        if code == 500:
            logging.exception('Request failed')
        return jsonify(error=str(exc) if code != 500 else 'Operation failed. See logs/app.log.'), code

    @app.get('/')
    def index():
        return render_template('index.html', token=token)

    @app.get('/api/state')
    def state():
        return jsonify(manager.snapshot())

    @app.post('/api/upload')
    def upload():
        options = json.loads(request.form.get('options', '{}'))
        files = request.files.getlist('images')
        if not files:
            raise ValueError('Choose at least one image')
        ids = [manager.add(file, options) for file in files]
        return jsonify(ids=ids)

    @app.post('/api/control')
    def control():
        data = request.get_json()
        manager.control(data['action'], data.get('ids'), data.get('options'))
        return jsonify(ok=True)

    @app.get('/api/thumbnail/<key>')
    def thumbnail(key):
        return send_file(manager.job_path(key, 'thumbnail'), mimetype='image/png')

    @app.get('/api/download/<key>')
    def download(key):
        return send_file(manager.job_path(key), as_attachment=True)

    @app.post('/api/delete/<key>')
    def delete(key):
        manager.delete_output(key)
        return jsonify(ok=True)

    @app.post('/api/open/<key>')
    def open_local(key):
        path = manager.job_path(key)
        if (request.get_json(silent=True) or {}).get('folder'):
            path = path.parent
        if sys.platform == 'win32':
            os.startfile(str(path))
        elif sys.platform == 'darwin':
            subprocess.Popen(['open', str(path)])
        else:
            subprocess.Popen(['xdg-open', str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return jsonify(ok=True)

    @app.post('/api/zip')
    def zip_start():
        manager.start_zip()
        return jsonify(ok=True)

    @app.get('/api/zip/download')
    def zip_download():
        with manager.lock:
            if manager.zip_state['status'] != 'Completed':
                raise ValueError('ZIP is not ready')
            path = manager.root / 'temp/exports' / f"{manager.zip_state['id']}.zip"
        return send_file(path, as_attachment=True, download_name='ETSY_UPSCALED_IMAGES.zip')

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Local Etsy AI Image Upscaler')
    parser.add_argument('--port', type=int, default=7860)
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    (ROOT / 'temp').mkdir(exist_ok=True)
    try:
        with FileLock(ROOT / 'temp/application.lock', timeout=0):
            application = create_app()
            print(f'Etsy AI Image Upscaler: http://127.0.0.1:{args.port}', flush=True)
            print('Keep this window open. Press Ctrl+C to stop. Outputs and queue are saved.', flush=True)
            if not args.no_browser:
                threading.Timer(1.2, lambda: webbrowser.open(f'http://127.0.0.1:{args.port}')).start()
            try:
                serve(application, host='127.0.0.1', port=args.port, threads=6)
            finally:
                application.extensions['queue'].close()
    except Timeout:
        print('This project is already running. Close its other CMD window first.')
        raise SystemExit(1)
    except OSError as exc:
        print(f'Cannot start the local server: {exc}. Try run.bat --port 7861.')
        raise SystemExit(1)
    except KeyboardInterrupt:
        print('Stopped. Queue and completed images are saved.')
