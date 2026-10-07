"""Format-preserving export; bounded JPEG search and lossless PNG."""
import io
from PIL import Image


def encode_image(image, fmt, dpi=False, compress=False, level='balanced', target_mb=None,
                 checkpoint=lambda: None, profile='standard'):
    fmt = fmt.upper().replace('JPG', 'JPEG')
    if fmt not in ('JPEG', 'PNG', 'WEBP') or profile not in ('standard', 'efficient'):
        raise ValueError('Unsupported output format or compression profile')
    if fmt == 'WEBP' and dpi:
        exif = Image.Exif()
        exif[282], exif[283], exif[296] = 300, 300, 2
        metadata = {'exif': exif.tobytes()}
    else:
        metadata = {'dpi': (300, 300)} if dpi else {}
    if image.info.get('icc_profile'):
        metadata['icc_profile'] = image.info['icc_profile']
    if fmt == 'JPEG' and image.mode == 'RGBA' and image.getchannel('A').getextrema()[0] < 255:
        raise ValueError('Transparent images cannot be exported as JPG. Choose PNG or WEBP.')
    if fmt == 'JPEG':
        image = image.convert('RGB')
    efficient = profile == 'efficient' and compress
    target = int(float(target_mb) * 1024 * 1024) if compress and target_mb else None
    floor = ({'light': 85, 'balanced': 72, 'maximum': 65} if efficient else
             {'light': 90, 'balanced': 85, 'maximum': 80})[level]

    def encode(quality):
        checkpoint()
        buf = io.BytesIO()
        options = dict(metadata)
        if fmt == 'PNG':
            options.update(optimize=compress and not efficient, compress_level=4 if efficient else 9 if compress else 6)
        elif fmt == 'JPEG':
            options.update(quality=quality, subsampling=2 if efficient else 0, optimize=True)
        else:
            options.update(quality=quality, method=2 if efficient else 6 if compress else 4)
        image.save(buf, format=fmt, **options)
        return buf.getvalue()

    if efficient and fmt != 'PNG':
        ceiling = 90
        data = encode(ceiling)
        if target and len(data) > target:
            # At most 7 further encodes; find the highest quality meeting the cap.
            data = encode(floor)
            if len(data) <= target:
                low, high = floor + 1, ceiling - 1
                while low <= high:
                    middle = (low + high) // 2
                    candidate = encode(middle)
                    if len(candidate) <= target:
                        data, low = candidate, middle + 1
                    else:
                        high = middle - 1
    else:
        qualities = list(range(96, floor - 1, -2)) if compress and fmt != 'PNG' else [98]
        if qualities[-1] > floor and compress and fmt != 'PNG':
            qualities.append(floor)
        for quality in qualities:
            data = encode(quality)
            if not target or len(data) <= target:
                break
    warning = 'Target size could not be reached without exceeding the quality floor.' if target and len(data) > target else ''
    return data, warning
