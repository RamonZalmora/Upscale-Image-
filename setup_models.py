"""Download and verify official model before the first batch."""
from pathlib import Path
import sys
from core.upscaler import Upscaler

if __name__ == '__main__':
    engine = Upscaler(Path(__file__).resolve().parent / 'models')
    try:
        path = engine.ensure_model(lambda p: print(f'\rDownloading Real-ESRGAN: {p}%', end='', flush=True))
        engine.load()
        print(f'\nModel verified: {path.name}. Device: {engine.device}')
        light_path = engine.ensure_model(lambda p: print(f'\rDownloading light FSRCNN: {p}%', end='', flush=True), model_name='light')
        engine.light.load(light_path)
        print(f'\nLight model verified: {light_path.name}. CPU threads: at most 2')
    except Exception as exc:
        print(f'\nModel setup failed: {exc}', file=sys.stderr)
        raise SystemExit(1)
