"""Format-preserving export with a visual quality floor and lossless PNG."""
import io
from PIL import Image


def encode_image(image, fmt, dpi=False, compress=False, level='balanced', target_mb=None, checkpoint=lambda: None):
    fmt = fmt.upper().replace('JPG', 'JPEG')
    if fmt not in ('JPEG', 'PNG', 'WEBP'):
        raise ValueError('Unsupported output format')
    if fmt == 'WEBP' and dpi:
        # Pillow embeds resolution in EXIF for WebP.
        exif = Image.Exif()
        exif[282] = 300
        exif[283] = 300
        exif[296] = 2
        metadata = {'exif': exif.tobytes()}
    else:
        metadata = {'dpi': (300, 300)} if dpi else {}
    if image.info.get('icc_profile'):
        metadata['icc_profile'] = image.info['icc_profile']
    if fmt == 'JPEG' and image.mode == 'RGBA' and image.getchannel('A').getextrema()[0] < 255:
        raise ValueError('Transparent images cannot be exported as JPG. Choose PNG or WEBP.')
    if fmt == 'JPEG':
        image = image.convert('RGB')
    target = int(float(target_mb) * 1024 * 1024) if compress and target_mb else None
    floor = {'light': 90, 'balanced': 85, 'maximum': 80}[level]
    qualities = list(range(96, floor - 1, -2)) if compress and fmt != 'PNG' else [98]
    if qualities[-1] > floor and compress and fmt != 'PNG':
        qualities.append(floor)
    data = None
    for quality in qualities:
        checkpoint()
        buf = io.BytesIO()
        options = dict(metadata)
        if fmt == 'PNG':
            options.update(optimize=compress, compress_level=9 if compress else 6)
        elif fmt == 'JPEG':
            options.update(quality=quality, subsampling=0, optimize=True)
        else:
            options.update(quality=quality, method=6 if compress else 4)
        image.save(buf, format=fmt, **options)
        data = buf.getvalue()
        if not target or len(data) <= target:
            break
    warning = ''
    if target and len(data) > target:
        warning = 'Target size could not be reached without exceeding the quality floor.'
    return data, warning
