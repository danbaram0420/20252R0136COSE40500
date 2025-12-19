import os
import time
import datetime
import numpy as np
import gradio as gr
import torch
from PIL import Image, ImageChops, ImageFilter
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import logging as dlogging

# ----------------------------
# Gradio/Hub telemetry off (UI/통신 오버헤드 최소화)
# ----------------------------
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
dlogging.set_verbosity_error()

# ----------------------------
# Model config
# ----------------------------
MODEL_ID = "ShinoharaHare/Waifu-Inpaint-XL"
HF_TOKEN = os.environ.get("HF_TOKEN")  # gated 모델이면 환경변수로 넣어두기

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

# 4090 기준: TF32/benchmark가 대체로 유리
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

# ----------------------------
# Load pipeline
# ----------------------------
# diffusers 0.36+: torch_dtype deprecated, dtype 권장
try:
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        use_safetensors=True,
        token=HF_TOKEN,
    )
except TypeError:
    # 혹시 dtype를 못 받는 구버전 diffusers면 torch_dtype로 fallback
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=dtype,
        use_safetensors=True,
        token=HF_TOKEN,
    )

pipe = pipe.to(device)
pipe.set_progress_bar_config(disable=True)  # tqdm/progress bar 비활성화 (UI/터미널 오버헤드 방지)

# (중요) 4090 환경에서는 slicing은 보통 "메모리 절약" 옵션이라 오히려 느릴 수 있음.
# pipe.enable_attention_slicing()
# pipe.enable_vae_slicing()

# channels_last는 종종 속도에 도움(환경 따라 무해)
try:
    pipe.unet.to(memory_format=torch.channels_last)
except Exception:
    pass


# ----------------------------
# Helpers: ImageEditor -> (image, mask)
# ----------------------------
def _editor_to_image_and_mask(editor_value, diff_thresh: int = 8):
    """
    Gradio ImageEditor의 editor_value(dict)에서:
      - background: 원본
      - composite: 원본 + 사용자가 칠한 결과
    두 이미지의 차이를 이용해 마스크를 안정적으로 생성한다.
    """
    if editor_value is None or not isinstance(editor_value, dict):
        raise gr.Error("이미지를 올리고 마스크를 그려주세요.")

    bg = editor_value.get("background", None)
    comp = editor_value.get("composite", None)

    if bg is None:
        raise gr.Error("배경 이미지를 올려주세요.")
    if not isinstance(bg, Image.Image):
        bg = Image.fromarray(bg)
    bg = bg.convert("RGB")

    if comp is None:
        # composite가 없으면 layers fallback (브라우저/버전에 따라)
        layers = editor_value.get("layers", None)
        if not layers:
            raise gr.Error("마스크를 브러시로 칠해 주세요.")
        layer = layers[-1]
        if not isinstance(layer, Image.Image):
            layer = Image.fromarray(layer)
        # layer 자체가 변화가 있는 픽셀만 추출되게 차이 계산
        diff = layer.convert("RGB")
    else:
        if not isinstance(comp, Image.Image):
            comp = Image.fromarray(comp)
        comp = comp.convert("RGB")
        diff = ImageChops.difference(comp, bg)

    diff_l = diff.convert("L")
    mask = diff_l.point(lambda p: 255 if p > diff_thresh else 0).convert("L")
    return bg, mask


def _mask_to_bbox(mask_l: Image.Image, margin: int, snap: int = 8):
    m = np.array(mask_l)
    ys, xs = np.where(m > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    W, H = mask_l.size

    x0 = max(0, x0 - margin)
    y0 = max(0, y0 - margin)
    x1 = min(W, x1 + margin)
    y1 = min(H, y1 + margin)

    def _expand_to_multiple(start, end, limit, k):
        size = end - start
        new_size = ((size + k - 1) // k) * k
        extra = new_size - size
        start2 = max(0, start - extra // 2)
        end2 = min(limit, start2 + new_size)
        start2 = max(0, end2 - new_size)
        return start2, end2

    x0, x1 = _expand_to_multiple(x0, x1, W, snap)
    y0, y1 = _expand_to_multiple(y0, y1, H, snap)
    return (x0, y0, x1, y1)


def _dilate_and_feather(mask_l: Image.Image, dilate_px: int = 0, feather_px: int = 0):
    m = mask_l
    if dilate_px > 0:
        m = m.filter(ImageFilter.MaxFilter(size=2 * dilate_px + 1))
    if feather_px > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=feather_px))
    return m


# ----------------------------
# Inpaint (최종 결과만 반환)
# ----------------------------
@torch.inference_mode()
def run_inpaint(editor_value, prompt, negative_prompt, steps, cfg, strength, seed,
                use_crop, margin, dilate_px, feather_px):
    image, mask = _editor_to_image_and_mask(editor_value)

    if seed is None or int(seed) < 0:
        gen = None
    else:
        gen = torch.Generator(device=pipe.device).manual_seed(int(seed))

    # crop 모드
    if use_crop:
        bbox = _mask_to_bbox(mask, margin=int(margin), snap=8)
        if bbox is None:
            raise gr.Error("마스크가 비어 있어요. 지울 영역을 칠해 주세요.")

        x0, y0, x1, y1 = bbox
        image_crop = image.crop((x0, y0, x1, y1))
        mask_crop = mask.crop((x0, y0, x1, y1))

        mask_crop_proc = _dilate_and_feather(mask_crop, dilate_px=int(dilate_px), feather_px=0)
        mask_for_blend = _dilate_and_feather(mask_crop, dilate_px=int(dilate_px), feather_px=int(feather_px))

        t0 = time.perf_counter()
        out_crop = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if str(negative_prompt).strip() else None,
            image=image_crop,
            mask_image=mask_crop_proc,
            num_inference_steps=int(steps),
            guidance_scale=float(cfg),
            strength=float(strength),
            height=image_crop.height,
            width=image_crop.width,
            generator=gen,
        ).images[0]
        t1 = time.perf_counter()

        blended_crop = Image.composite(out_crop, image_crop, mask_for_blend.convert("L"))
        result = image.copy()
        result.paste(blended_crop, (x0, y0))

        info = f"done (crop {image_crop.size} / full {image.size}) in {t1 - t0:.2f}s"
        return result, info

    # full 모드 (test.py와 유사하게 height/width 명시)
    t0 = time.perf_counter()
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if str(negative_prompt).strip() else None,
        image=image,
        mask_image=_dilate_and_feather(mask, dilate_px=int(dilate_px), feather_px=0),
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        strength=float(strength),
        height=image.height,
        width=image.width,
        generator=gen,
    ).images[0]
    t1 = time.perf_counter()
    info = f"done (full {image.size}) in {t1 - t0:.2f}s"
    return out, info


def save_result(img: Image.Image):
    if img is None:
        raise gr.Error("저장할 결과가 없어요. 먼저 Inpaint를 실행하세요.")

    out_dir = "outputs"
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"inpaint_{ts}.png")
    img.save(path)
    return path, f"saved: {path}"


# ----------------------------
# UI
# ----------------------------
with gr.Blocks(analytics_enabled=False) as demo:
    gr.Markdown("# Waifu-Inpaint-XL Inpainting GUI (최종 결과만 표시 + 저장 버튼)")

    with gr.Row():
        inp = gr.ImageEditor(label="원본 + 마스크(브러시로 칠하기)", type="pil")
        out_img = gr.Image(label="결과(최종)", type="pil")

    prompt = gr.Textbox(label="Prompt", value="remove text, clean background, anime style", lines=2)
    negative = gr.Textbox(label="Negative Prompt", value="low quality, blurry, artifacts", lines=2)

    with gr.Row():
        steps = gr.Slider(10, 60, value=28, step=1, label="Steps")
        cfg = gr.Slider(1.0, 12.0, value=5.0, step=0.5, label="CFG (Guidance)")
        strength = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Denoise/Strength")
        seed = gr.Number(value=5, label="Seed (-1이면 랜덤)")

    with gr.Accordion("고급 설정(속도/품질)", open=False):
        use_crop = gr.Checkbox(value=True, label="Crop 모드(마스크 주변만 inpaint → 원본 합성)  ※ 텍스트 제거에 강력 추천")
        margin = gr.Slider(32, 384, value=128, step=8, label="Crop margin(px)")
        dilate_px = gr.Slider(0, 16, value=8, step=1, label="Mask dilation(px) (텍스트 잔상 줄이기)")
        feather_px = gr.Slider(0, 16, value=4, step=1, label="Feather(px) (경계 부드럽게)")

    status = gr.Textbox(label="Status", value="", interactive=False)

    with gr.Row():
        run = gr.Button("Inpaint", variant="primary")
        save = gr.Button("Save result", variant="secondary")

    # 저장 결과: 파일로 다운로드 제공
    out_file = gr.File(label="Saved file (download)")
    save_status = gr.Textbox(label="Save status", value="", interactive=False)

    run.click(
        fn=run_inpaint,
        inputs=[inp, prompt, negative, steps, cfg, strength, seed, use_crop, margin, dilate_px, feather_px],
        outputs=[out_img, status],
    )

    save.click(
        fn=save_result,
        inputs=[out_img],
        outputs=[out_file, save_status],
    )

# 참고: share=True는 네트워크/브라우저 통신 부하가 늘 수 있어서 기본은 False
demo.launch()
