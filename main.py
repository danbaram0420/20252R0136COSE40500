import gradio as gr
import torch
from diffusers import StableDiffusionXLInpaintPipeline
from PIL import Image

MODEL_ID = "ShinoharaHare/Waifu-Inpaint-XL"

device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=dtype,
    use_safetensors=True,
)
pipe = pipe.to(device)

# 선택: VRAM 절약/속도 옵션 (환경에 따라)
# pipe.enable_xformers_memory_efficient_attention()  # xformers 설치 시
# pipe.enable_model_cpu_offload()  # VRAM이 작으면 유용

def inpaint(sketch, prompt, negative_prompt, steps, cfg, strength, seed):
    """
    sketch는 gr.Image(tool='sketch')로 들어오면 {"image": PIL/np, "mask": PIL/np} 형태
    (Gradio가 tool=sketch일 때 dict로 넘겨주는 동작) :contentReference[oaicite:3]{index=3}
    """
    if sketch is None or not isinstance(sketch, dict) or "image" not in sketch or "mask" not in sketch:
        raise gr.Error("이미지를 올리고 마스크를 그려주세요. (왼쪽 이미지에서 브러시로 칠하면 됩니다)")

    image = sketch["image"]
    mask = sketch["mask"]

    # Gradio가 numpy로 줄 때도 있으니 PIL로 통일
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if not isinstance(mask, Image.Image):
        mask = Image.fromarray(mask)

    # mask는 보통 흑백(또는 RGBA)로 들어올 수 있음 → L로 변환
    mask = mask.convert("L")

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
        inp = gr.Image(label="원본 + 마스크(브러시로 칠하기)", tool="sketch", type="pil")
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
