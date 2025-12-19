import math
import numpy as np
from PIL import Image

def _bbox_from_mask(mask_l: Image.Image, thresh: int = 1):
    """mask(L)에서 흰 영역 bbox 추출. 없으면 None."""
    m = np.array(mask_l)
    ys, xs = np.where(m >= thresh)
    if len(xs) == 0:
        return None
    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    return x0, y0, x1, y1

def _expand_bbox(x0, y0, x1, y1, W, H, margin_px):
    x0 = max(0, x0 - margin_px)
    y0 = max(0, y0 - margin_px)
    x1 = min(W, x1 + margin_px)
    y1 = min(H, y1 + margin_px)
    return x0, y0, x1, y1

def _pad_to_multiple_of_8(x0, y0, x1, y1, W, H):
    """
    SDXL latent/grid 때문에 ROI 사이즈가 8의 배수면 안정적임.
    bbox를 밖으로 확장해서 w,h를 8 배수로 맞춤.
    """
    w = x1 - x0
    h = y1 - y0
    w2 = int(math.ceil(w / 8) * 8)
    h2 = int(math.ceil(h / 8) * 8)

    dx = w2 - w
    dy = h2 - h

    # 좌우/상하로 반반 확장 (경계는 clamp)
    x0n = max(0, x0 - dx // 2)
    y0n = max(0, y0 - dy // 2)
    x1n = min(W, x0n + w2)
    y1n = min(H, y0n + h2)

    # 혹시 clamp 때문에 크기가 줄었으면 반대쪽 보정
    x0n = max(0, x1n - w2)
    y0n = max(0, y1n - h2)
    return x0n, y0n, x1n, y1n

def crop_inpaint_paste(
    pipe,
    image_rgb: Image.Image,
    mask_l: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    steps: int = 28,
    guidance: float = 5.0,
    seed: int = 5,
    margin_px: int = 64,
):
    """
    - mask 흰 영역 bbox + margin을 ROI로 잡고
    - ROI만 inpaint 수행
    - 결과를 원본에 '마스크 흰 영역만' 덮어씀 (마스크 밖은 원본 고정)
    """
    image_rgb = image_rgb.convert("RGB")
    mask_l = mask_l.convert("L")
    W, H = image_rgb.size

    bbox = _bbox_from_mask(mask_l, thresh=1)
    if bbox is None:
        # 마스크가 비어있으면 그냥 원본 반환
        return image_rgb

    x0, y0, x1, y1 = bbox
    x0, y0, x1, y1 = _expand_bbox(x0, y0, x1, y1, W, H, margin_px)
    x0, y0, x1, y1 = _pad_to_multiple_of_8(x0, y0, x1, y1, W, H)

    # ROI crop
    roi_img = image_rgb.crop((x0, y0, x1, y1))
    roi_msk = mask_l.crop((x0, y0, x1, y1))

    # generator
    import torch
    gen = torch.Generator(device="cuda").manual_seed(int(seed))

    # ROI inpaint
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        roi_out = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            image=roi_img,
            mask_image=roi_msk,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=roi_img.width,
            height=roi_img.height,
            generator=gen,
        ).images[0]

    # ✅ “마스크 영역만” 원본에 합성 (마스크 밖 픽셀은 원본 유지)
    out = image_rgb.copy()

    # roi_out을 일단 붙이고, 그 위에서 마스크로 합성
    base_patch = out.crop((x0, y0, x1, y1)).convert("RGB")
    roi_out = roi_out.convert("RGB")

    # Image.composite(A,B,mask): mask 흰 부분은 A, 검정은 B
    blended_patch = Image.composite(roi_out, base_patch, roi_msk)

    out.paste(blended_patch, (x0, y0))
    return out
