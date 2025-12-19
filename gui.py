import time
import numpy as np
import torch
import gradio as gr
from PIL import Image
from diffusers import StableDiffusionXLInpaintPipeline

MODEL_ID = "ShinoharaHare/Waifu-Inpaint-XL"

# ----------------------------
# 0) 전역 1회 로드 (콜백 안에서 from_pretrained 금지)
# ----------------------------
def load_pipe():
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # diffusers 0.36.0에서 dtype=는 무시될 수 있으니 torch_dtype로 확실히 fp16 강제
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
    ).to("cuda")

    pipe.set_progress_bar_config(disable=True)

    # xformers 있으면 켜기(없으면 무시)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # 로드 상태 확인 로그(중요)
    print("[pipe] unet device:", next(pipe.unet.parameters()).device)
    print("[pipe] unet dtype :", next(pipe.unet.parameters()).dtype)

    return pipe

PIPE = load_pipe()

# ----------------------------
# 1) 유틸
# ----------------------------
def _to_pil_rgba(x):
    if x is None:
        return None
    if isinstance(x, Image.Image):
        return x.convert("RGBA")
    return Image.fromarray(x).convert("RGBA")

def _round8(v: int) -> int:
    v = int(v)
    return max(8, (v // 8) * 8)

def _make_mask_from_editor_payload(payload, thresh=8):
    """
    Gradio ImageEditor 결과(dict)에서:
      - background(원본)과 composite(그린 결과)의 픽셀 차이로 마스크 생성
    """
    if payload is None:
        return None, None

    if isinstance(payload, dict):
        bg = payload.get("background", None)
        comp = payload.get("composite", None)

        if bg is None or comp is None:
            return None, None

        bg = _to_pil_rgba(bg)
        comp = _to_pil_rgba(comp)

        bg_np = np.array(bg).astype(np.int16)
        comp_np = np.array(comp).astype(np.int16)

        diff = np.abs(comp_np - bg_np)  # (H,W,4)
        diff_mag = diff[:, :, :3].sum(axis=2) + diff[:, :, 3]
        mask = (diff_mag > thresh).astype(np.uint8) * 255

        image = bg.convert("RGB")
        mask_pil = Image.fromarray(mask).convert("L")
        return image, mask_pil

    return None, None

# ----------------------------
# 2) 추론 함수 (Run 버튼에서만)
# ----------------------------
@torch.inference_mode()
def run_inpaint(editor_payload, prompt, negative_prompt, steps, guidance, seed,
                use_original_size, out_w, out_h, show_mask_preview):
    image, mask = _make_mask_from_editor_payload(editor_payload)

    if image is None:
        raise gr.Error("이미지를 업로드해 주세요.")
    if mask is None:
        raise gr.Error("마스크가 없습니다. (왼쪽에서 브러시로 칠해 주세요.)")

    # 크기 결정
    if use_original_size:
        width = _round8(image.width)
        height = _round8(image.height)
    else:
        width = _round8(out_w)
        height = _round8(out_h)

    # 추론 직전 1회 리사이즈
    if (image.width, image.height) != (width, height):
        image = image.resize((width, height), Image.BICUBIC)
    if (mask.width, mask.height) != (width, height):
        mask = mask.resize((width, height), Image.NEAREST)

    seed = int(seed)
    gen = torch.Generator(device="cuda").manual_seed(seed)

    t0 = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.float16):
        out = PIPE(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            image=image,
            mask_image=mask,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            height=height,
            width=width,
            generator=gen,
        ).images[0]
    dt = time.perf_counter() - t0

    info = (
        f"done | {width}x{height} | steps={steps} | cfg={guidance} | seed={seed} | "
        f"time={dt:.2f}s"
    )

    if show_mask_preview:
        return out, mask, info
    else:
        return out, None, info

# ----------------------------
# 3) UI (Gradio 6.1.0 호환: ImageEditor 사용)
# ----------------------------
with gr.Blocks(title="Waifu-Inpaint-XL (Fast GUI, Gradio 6.1.0)") as demo:
    gr.Markdown(
        "### Waifu-Inpaint-XL Inpaint GUI\n"
        "- **왼쪽에서 이미지 업로드 후 브러시로 칠한 영역**이 인페인트 영역입니다.\n"
        "- 추론은 **Run 버튼**에서만 실행됩니다.\n"
    )

    with gr.Row():
        with gr.Column(scale=1):
            editor = gr.ImageEditor(
                label="Input + Paint Mask",
                type="pil",
                height=512,
            )

            prompt = gr.Textbox(label="Prompt", value="blue eyes, holding red spider lily in hand")
            negative_prompt = gr.Textbox(label="Negative Prompt (optional)", value="")

            with gr.Row():
                steps = gr.Slider(1, 60, value=28, step=1, label="Steps")
                guidance = gr.Slider(0.0, 20.0, value=5.0, step=0.5, label="Guidance (CFG)")

            with gr.Row():
                seed = gr.Number(value=5, precision=0, label="Seed")
                use_original_size = gr.Checkbox(value=True, label="Use original size (추천)")

            with gr.Row():
                out_w = gr.Number(value=1024, precision=0, label="Width (original 미사용 시)")
                out_h = gr.Number(value=1024, precision=0, label="Height (original 미사용 시)")

            show_mask_preview = gr.Checkbox(value=True, label="Show derived mask preview")

            run_btn = gr.Button("Run Inpaint", variant="primary")

        with gr.Column(scale=1):
            out_img = gr.Image(label="Output", type="pil", height=512)
            mask_preview = gr.Image(label="Mask Preview (white=inpaint)", type="pil", height=512)
            out_info = gr.Textbox(label="Info / Timing", interactive=False)

    # ✅ Gradio 5+/6+ 방식: 이벤트에 concurrency_limit 걸기 (동시 추론 방지)
    run_btn.click(
        fn=run_inpaint,
        inputs=[editor, prompt, negative_prompt, steps, guidance, seed,
                use_original_size, out_w, out_h, show_mask_preview],
        outputs=[out_img, mask_preview, out_info],
        concurrency_limit=1,
    )

# ✅ queue에는 concurrency_count가 없음 → max_size + (가능하면 default_concurrency_limit)
try:
    demo.queue(max_size=8, default_concurrency_limit=1)
except TypeError:
    demo.queue(max_size=8)

# (선택) max_threads는 launch에 있는 경우가 많지만, 버전별로 다를 수 있어 try로 감쌈
try:
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, max_threads=1)
except TypeError:
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
