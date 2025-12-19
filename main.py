import math
import numpy as np
from PIL import Image, ImageFilter
import torch
from diffusers import StableDiffusionXLInpaintPipeline

MODEL_ID = "ShinoharaHare/Waifu-Inpaint-XL"

# ----------------------------
# Utils: pad to multiple of 8 (edge pad for image, zero pad for mask)
# ----------------------------
def pad_to_multiple_of_8(image_rgb: Image.Image, mask_l: Image.Image):
    image_rgb = image_rgb.convert("RGB")
    mask_l = mask_l.convert("L")

    W, H = image_rgb.size
    W8 = int(math.ceil(W / 8) * 8)
    H8 = int(math.ceil(H / 8) * 8)

    pad_r = W8 - W
    pad_b = H8 - H
    if pad_r == 0 and pad_b == 0:
        return image_rgb, mask_l, (0, 0, W, H)  # no pad, crop box is full

    # image: edge padding
    img = np.array(image_rgb)
    img_pad = np.pad(img, ((0, pad_b), (0, pad_r), (0, 0)), mode="edge")
    image_padded = Image.fromarray(img_pad)

    # mask: zero padding (outside = keep original)
    m = np.array(mask_l)
    m_pad = np.pad(m, ((0, pad_b), (0, pad_r)), mode="constant", constant_values=0)
    mask_padded = Image.fromarray(m_pad).convert("L")

    crop_box = (0, 0, W, H)  # later crop back to original
    return image_padded, mask_padded, crop_box


# ----------------------------
# Feather mask: dilate + blur (for smooth boundary)
# ----------------------------
def feather_mask(mask_l: Image.Image, dilate_px=8, blur_px=8):
    """
    mask_l: L mode, white=inpaint region
    dilate_px: expand mask to cover boundary artifacts
    blur_px: soften edge for seamless blending
    """
    m = mask_l.convert("L")

    if dilate_px and dilate_px > 0:
        # MaxFilter kernel size must be odd
        k = 2 * int(dilate_px) + 1
        m = m.filter(ImageFilter.MaxFilter(size=k))

    if blur_px and blur_px > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=float(blur_px)))

    return m


# ----------------------------
# Core: full inpaint -> composite onto original using feathered mask
# ----------------------------
@torch.inference_mode()
def full_inpaint_then_composite(
    pipe: StableDiffusionXLInpaintPipeline,
    image_rgb: Image.Image,
    mask_l: Image.Image,
    prompt: str,
    negative_prompt: str = "",
    steps: int = 28,
    guidance: float = 5.0,
    seed: int = 5,
    dilate_px: int = 8,
    blur_px: int = 8,
):
    """
    1) pad image/mask to multiple-of-8 (SDXL friendly)
    2) full-image inpaint (model sees full context, tends to be best)
    3) crop back to original size
    4) composite: only masked region applied, outside forced to original
       (feathered mask to avoid seams)
    """
    image_rgb = image_rgb.convert("RGB")
    mask_l = mask_l.convert("L")

    # safety: if mask size != image size, force resize
    if mask_l.size != image_rgb.size:
        mask_l = mask_l.resize(image_rgb.size, Image.NEAREST)

    # pad to multiple of 8
    img_p, msk_p, crop_box = pad_to_multiple_of_8(image_rgb, mask_l)

    # generator
    gen = torch.Generator(device="cuda").manual_seed(int(seed))

    # full inpaint
    with torch.autocast("cuda", dtype=torch.float16):
        gen_img = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            image=img_p,
            mask_image=msk_p,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=img_p.width,
            height=img_p.height,
            generator=gen,
        ).images[0].convert("RGB")

    # crop back to original size if padded
    gen_img = gen_img.crop(crop_box)

    # feather mask for blending (on original resolution)
    fm = feather_mask(mask_l, dilate_px=dilate_px, blur_px=blur_px)

    # composite: white->generated, black->original
    out = Image.composite(gen_img, image_rgb, fm)
    return out


# ----------------------------
# Example main
# ----------------------------
def main():
    # 1) load pipeline once
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # 2) inputs
    image = Image.open("images/sample.jpg").convert("RGB")
    mask  = Image.open("masks/sample_mask.png").convert("L")  # white=inpaint

    # 프롬프트는 “로고 제거”면 배경 묘사/부정 프롬프트가 중요함
    prompt = "clean natural background with tree"
    negative = "logo, watermark, text, emblem, symbol, icon, letters"

    # 3) run
    out = full_inpaint_then_composite(
        pipe,
        image,
        mask,
        prompt=prompt,
        negative_prompt=negative,
        steps=28,
        guidance=5.0,
        seed=5,
        dilate_px=10,   # 경계 흔들림 방지용(로고 제거에 보통 유리)
        blur_px=10,     # seam 부드럽게
    )

    out.save("out_full_inpaint_composite.png")
    print("Saved: out_full_inpaint_composite.png")


if __name__ == "__main__":
    main()
