import re
from pathlib import Path
from PIL import Image, ImageOps

SCALES = (2, 2.5, 3, 3.5, 4, 4.5)


def output_dimensions(width, height, scale):
    if isinstance(scale, bool) or scale not in SCALES:
        raise ValueError('Invalid upscale scale')
    # Round half pixels up, matching JavaScript Math.round.
    return int(width * scale + .5), int(height * scale + .5)


def check_dimensions(width, height, scale, model_name):
    dimensions = output_dimensions(width, height, scale)
    native = 2 if model_name == 'light' else 4
    if max(width * height * native**2, dimensions[0] * dimensions[1]) > 100_000_000:
        raise ValueError('AI or final output exceeds the safe 100 megapixel limit.')
    return dimensions


SUPPORTED = {'.jpg', '.jpeg', '.png', '.webp'}
Image.MAX_IMAGE_PIXELS = 25_000_000


def inspect_image(path, thumbnail):
    with Image.open(path) as image:
        if image.format not in ('JPEG', 'PNG', 'WEBP'):
            raise ValueError('Supported formats: JPG, PNG, WEBP')
        image.verify()
    with Image.open(path) as image:
        fmt = image.format
        width, height = image.size
        if image.getexif().get(274) in (5, 6, 7, 8):
            width, height = height, width
        if width * height > 25_000_000:
            raise ValueError('Input exceeds the safe 25 megapixel limit.')
        alpha = image.mode in ('RGBA', 'LA') or 'transparency' in image.info
        # JPEG uses decoder-side reduction; PNG/WebP decoders still need a pixel decode.
        image.draft('RGB', (288, 288))
        image = ImageOps.exif_transpose(image)
        image.thumbnail((144, 100), Image.Resampling.LANCZOS)
        image.convert('RGBA').save(thumbnail, 'PNG')
    return width, height, fmt, alpha


def safe_name(value):
    value = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '-', str(value)).strip(' .')[:100]
    if not value:
        value = 'image'
    if value.split('.')[0].upper() in {'CON', 'PRN', 'AUX', 'NUL', *[f'COM{i}' for i in range(1, 10)], *[f'LPT{i}' for i in range(1, 10)]}:
        value = '_' + value
    return value


def reserve_output(folder, stem, suffix):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    index = 0
    while True:
        candidate = folder / f'{stem}{"" if index == 0 else f"-{index:03d}"}.{suffix}'
        try:
            with candidate.open('xb'):
                pass
            return candidate
        except FileExistsError:
            index += 1
