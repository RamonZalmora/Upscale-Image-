"""300 PPI guidance for common paper sizes; never crop or stretch artwork."""
from .image_utils import SCALES, output_dimensions

PAPERS = {
    'a4': ('A4', 210 / 25.4, 297 / 25.4, 2480, 3508),
    'letter': ('US Letter', 8.5, 11, 2550, 3300),
}


def print_guidance(width, height, paper, scale=1):
    if paper == 'none' or not width or not height:
        return None
    name, inches_w, inches_h, target_w, target_h = PAPERS[paper]
    if width > height:
        inches_w, inches_h, target_w, target_h = inches_h, inches_w, target_h, target_w
    output_w, output_h = (width, height) if scale == 1 else output_dimensions(width, height, scale)
    recommended = next((factor for factor in (1, *SCALES)
                        if (width if factor == 1 else output_dimensions(width, height, factor)[0]) >= target_w
                        and (height if factor == 1 else output_dimensions(width, height, factor)[1]) >= target_h), None)
    mismatch = abs((width / height) / (inches_w / inches_h) - 1) > .02
    return dict(paper=name, target_width=target_w, target_height=target_h,
                effective_ppi=round(min(output_w / inches_w, output_h / inches_h), 1),
                sufficient=output_w >= target_w and output_h >= target_h,
                recommended_scale=recommended, aspect_mismatch=mismatch)
