import gradio as gr
import torch
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image
import os
import numpy as np

MODEL_ID = "ShinoharaHare/Waifu-Inpaint-XL"
token = os.environ.get("HF_TOKEN")  # HF_TOKEN을 환경변수로 설정해둔 경우

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    use_safetensors=True,
    token=token,
).to("cuda")

# 선택: VRAM 절약/속도 옵션 (환경에 따라)
# pipe.enable_xformers_memory_efficient_attention()  # xformers 설치 시
# pipe.enable_model_cpu_offload()  # VRAM이 작으면 유용

def _editor_to_image_and_mask(editor_value):
    """
    gr.ImageEditor 입력(editor_value)에서:
    - background: 원본 이미지
    - layers[-1]의 alpha: 마스크
    를 추출해서 (image, mask[L])로 반환
    """
    if editor_value is None or not isinstance(editor_value, dict):
        raise gr.Error("이미지를 올리고 마스크를 그려주세요.")

    bg = editor_value.get("background", None)
    layers = editor_value.get("layers", None)

    if bg is None:
        raise gr.Error("배경 이미지를 올려주세요.")
    if not isinstance(bg, Image.Image):
        bg = Image.fromarray(bg)

    if not layers or len(layers) == 0:
        raise gr.Error("마스크를 브러시로 칠해 주세요.")

    # 보통 마지막 레이어가 사용자가 칠한 stroke
    layer = layers[-1]
    if not isinstance(layer, Image.Image):
        layer = Image.fromarray(layer)

    layer = layer.convert("RGBA")
    alpha = layer.split()[-1]  # A 채널

    # alpha > 0인 픽셀을 흰색(255), 아니면 검정(0)으로
    mask = alpha.point(lambda p: 255 if p > 0 else 0).convert("L")

    return bg.convert("RGB"), mask


def inpaint(editor_value, prompt, negative_prompt, steps, cfg, strength, seed):
    image, mask = _editor_to_image_and_mask(editor_value)

    generator = None
    if seed is not None and int(seed) >= 0:
        generator = torch.Generator(device=device).manual_seed(int(seed))

    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt.strip() else None,
        image=image,
        mask_image=mask,
        num_inference_steps=int(steps),
        guidance_scale=float(cfg),
        strength=float(strength),
        generator=generator,
    ).images[0]

    return out

with gr.Blocks() as demo:
    gr.Markdown("# Waifu-Inpaint-XL Inpainting GUI (로컬)")

    with gr.Row():
        inp = gr.ImageEditor(label="원본 + 마스크(브러시로 칠하기)", type="pil")
        out = gr.Image(label="결과", type="pil")

    prompt = gr.Textbox(label="Prompt", value="remove text, clean background, anime style", lines=2)
    negative = gr.Textbox(label="Negative Prompt", value="low quality, blurry, artifacts", lines=2)

    with gr.Row():
        steps = gr.Slider(10, 60, value=30, step=1, label="Steps")
        cfg = gr.Slider(1.0, 12.0, value=6.0, step=0.5, label="CFG (Guidance)")
        strength = gr.Slider(0.1, 1.0, value=0.8, step=0.05, label="Denoise/Strength")

    seed = gr.Number(value=-1, label="Seed (-1이면 랜덤)")

    run = gr.Button("Inpaint")
    run.click(
        fn=inpaint,
        inputs=[inp, prompt, negative, steps, cfg, strength, seed],
        outputs=[out],
    )

demo.launch()
