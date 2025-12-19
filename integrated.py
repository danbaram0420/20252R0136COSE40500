import os, time, math, tempfile
import numpy as np
from PIL import Image, ImageFilter
import torch
import gradio as gr
from diffusers import StableDiffusionXLInpaintPipeline

MODEL_ID = "ShinoharaHare/Waifu-Inpaint-XL"

# --------------------------
# 0) Pipeline: 전역 1회 로드
# --------------------------
def load_pipe():
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # diffusers 버전별 dtype/torch_dtype 호환
    try:
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            MODEL_ID, dtype=torch.float16
        )
    except TypeError:
        pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.float16
        )

    pipe = pipe.to("cuda")
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    print("[pipe] device:", next(pipe.unet.parameters()).device)
    print("[pipe] dtype :", next(pipe.unet.parameters()).dtype)
    return pipe

PIPE = load_pipe()

# --------------------------
# 1) Utils
# --------------------------
def pad_to_multiple_of_8(image_rgb: Image.Image, mask_l: Image.Image):
    W, H = image_rgb.size
    W8 = int(math.ceil(W / 8) * 8)
    H8 = int(math.ceil(H / 8) * 8)
    pad_r = W8 - W
    pad_b = H8 - H
    if pad_r == 0 and pad_b == 0:
        return image_rgb, mask_l, (0, 0, W, H)

    img = np.array(image_rgb)
    img_pad = np.pad(img, ((0, pad_b), (0, pad_r), (0, 0)), mode="edge")
    image_padded = Image.fromarray(img_pad)

    m = np.array(mask_l)
    m_pad = np.pad(m, ((0, pad_b), (0, pad_r)), mode="constant", constant_values=0)
    mask_padded = Image.fromarray(m_pad).convert("L")

    return image_padded, mask_padded, (0, 0, W, H)

def feather_mask(mask_l: Image.Image, dilate_px=10, blur_px=10):
    m = mask_l.convert("L")
    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        m = m.filter(ImageFilter.MaxFilter(size=k))
    if blur_px > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=float(blur_px)))
    return m

def _to_pil_rgba(x):
    if x is None: return None
    if isinstance(x, Image.Image): return x.convert("RGBA")
    if isinstance(x, str): return Image.open(x).convert("RGBA")
    return Image.fromarray(x).convert("RGBA")

def _mask_from_layers(layers, size_wh):
    W, H = size_wh
    acc = np.zeros((H, W), dtype=np.uint8)
    if not isinstance(layers, (list, tuple)):
        layers = [layers]
    for lay in layers:
        if lay is None:
            continue
        lay = _to_pil_rgba(lay)
        if lay.size != (W, H):
            lay = lay.resize((W, H), Image.NEAREST)
        a = np.array(lay)[:, :, 3]
        acc = np.maximum(acc, a)
    return (acc > 0).astype(np.uint8) * 255

def _mask_from_diff(bg_rgba, comp_rgba, thresh=8):
    bg = np.array(bg_rgba).astype(np.int16)
    cp = np.array(comp_rgba).astype(np.int16)
    diff = np.abs(cp - bg)
    diff_mag = diff[:, :, :3].sum(axis=2) + diff[:, :, 3]
    return (diff_mag > int(thresh)).astype(np.uint8) * 255

def extract_mask_from_editor(editor_payload, diff_thresh=8):
    """
    editor_payload(dict):
      - background/composite/layers 가 filepath 또는 PIL/np로 들어올 수 있음
    returns: (preview_rgb, preview_mask_l)
    """
    if editor_payload is None or not isinstance(editor_payload, dict):
        raise gr.Error("이미지를 업로드하고 마스크를 칠해 주세요.")

    bg = editor_payload.get("background", None)
    comp = editor_payload.get("composite", None)
    layers = editor_payload.get("layers", None)

    if bg is None:
        raise gr.Error("background가 없습니다. (업로드 후 다시 시도)")

    bg_rgba = _to_pil_rgba(bg)
    W, H = bg_rgba.size

    mask_np = None
    if layers is not None:
        try:
            mask_np = _mask_from_layers(layers, (W, H))
        except Exception:
            mask_np = None

    if mask_np is None:
        if comp is None:
            raise gr.Error("composite가 없어 마스크 생성 불가 (ImageEditor 이슈)")
        comp_rgba = _to_pil_rgba(comp)
        if comp_rgba.size != (W, H):
            comp_rgba = comp_rgba.resize((W, H), Image.NEAREST)
        mask_np = _mask_from_diff(bg_rgba, comp_rgba, thresh=diff_thresh)

    return bg_rgba.convert("RGB"), Image.fromarray(mask_np).convert("L")

def is_mask_empty(mask_l):
    return (np.array(mask_l) > 0).sum() == 0

def make_preview_image(orig_path, max_side=1024):
    """원본을 프리뷰용으로 축소해서 임시 파일로 저장, 경로 반환"""
    img = Image.open(orig_path).convert("RGB")
    W, H = img.size
    scale = min(1.0, max_side / max(W, H))
    if scale < 1.0:
        w2 = int(W * scale)
        h2 = int(H * scale)
        img2 = img.resize((w2, h2), Image.LANCZOS)
    else:
        img2 = img

    tmpdir = os.path.join(tempfile.gettempdir(), "waifu_inpaint_gui")
    os.makedirs(tmpdir, exist_ok=True)
    preview_path = os.path.join(tmpdir, f"preview_{int(time.time()*1000)}.png")
    img2.save(preview_path)
    return preview_path, (W, H), img2.size

@torch.inference_mode()
def full_inpaint_then_composite(pipe, image_rgb, mask_l, prompt, negative_prompt,
                                steps, guidance, seed, dilate_px, blur_px):
    image_rgb = image_rgb.convert("RGB")
    mask_l = mask_l.convert("L")
    img_p, msk_p, crop_box = pad_to_multiple_of_8(image_rgb, mask_l)

    gen = torch.Generator(device="cuda").manual_seed(int(seed))
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

    gen_img = gen_img.crop(crop_box)
    fm = feather_mask(mask_l, dilate_px=int(dilate_px), blur_px=int(blur_px))
    out = Image.composite(gen_img, image_rgb, fm)
    return out

# --------------------------
# 2) Callbacks (핵심: 업로드 때만 프리뷰 만들고, Run 때만 추론)
# --------------------------
def on_upload(orig_file, max_side):
    if orig_file is None:
        raise gr.Error("이미지를 업로드해 주세요.")
    # orig_file: filepath (gr.Image type="filepath")
    preview_path, orig_size, preview_size = make_preview_image(orig_file, max_side=int(max_side))
    state = {
        "orig_path": orig_file,
        "orig_size": orig_size,
        "preview_path": preview_path,
        "preview_size": preview_size,
    }
    # editor에 프리뷰 이미지를 세팅(배경)
    return preview_path, state, f"Loaded: {orig_size[0]}x{orig_size[1]}  |  Preview: {preview_size[0]}x{preview_size[1]}"

def run(editor_payload, state, prompt, negative_prompt,
        steps, guidance, seed, diff_thresh, dilate_px, blur_px,
        save_dir, basename, autosave, save_input_mask):

    t0 = time.perf_counter()

    if state is None or "orig_path" not in state:
        raise gr.Error("먼저 이미지를 업로드해 주세요.")

    orig_path = state["orig_path"]
    orig_size = tuple(state["orig_size"])

    # (A) 프리뷰에서 마스크 추출
    tA0 = time.perf_counter()
    preview_rgb, preview_mask = extract_mask_from_editor(editor_payload, diff_thresh=int(diff_thresh))
    if is_mask_empty(preview_mask):
        raise gr.Error("마스크가 비어있습니다. 칠한 영역이 없습니다.")
    tA = time.perf_counter() - tA0

    # (B) 마스크를 원본 크기로 리사이즈(NEAREST)
    mask_full = preview_mask.resize(orig_size, Image.NEAREST)

    # (C) 원본 로드 + full inpaint + composite
    tB0 = time.perf_counter()
    image_full = Image.open(orig_path).convert("RGB")
    out = full_inpaint_then_composite(
        PIPE, image_full, mask_full,
        prompt=prompt, negative_prompt=negative_prompt,
        steps=int(steps), guidance=float(guidance), seed=int(seed),
        dilate_px=int(dilate_px), blur_px=int(blur_px)
    )
    tB = time.perf_counter() - tB0
    t_all = time.perf_counter() - t0

    saved = []
    if autosave:
        save_dir = (save_dir or "outputs").strip()
        os.makedirs(save_dir, exist_ok=True)
        base = (basename or "result").strip()

        out_path = os.path.join(save_dir, f"{base}_out.png")
        out.save(out_path)
        saved.append(out_path)

        if save_input_mask:
            in_path = os.path.join(save_dir, f"{base}_input.png")
            mk_path = os.path.join(save_dir, f"{base}_mask.png")
            image_full.save(in_path)
            mask_full.save(mk_path)
            saved.extend([in_path, mk_path])

    info = (
        f"mask_extract(preview)={tA:.3f}s | inpaint+composite(full)={tB:.3f}s | total={t_all:.3f}s\n"
        f"orig={orig_size[0]}x{orig_size[1]} | steps={steps} cfg={guidance} seed={seed}\n"
        f"saved: {', '.join(saved) if saved else '(none)'}"
    )
    return out, mask_full, info

# --------------------------
# 3) UI
# --------------------------
with gr.Blocks(title="Waifu-Inpaint-XL Integrated (Fast Upload)") as demo:
    gr.Markdown(
        "### 통합 Inpaint GUI (프리뷰 마스크 + 전체 inpaint composite)\n"
        "- 업로드 시 **프리뷰(축소본)** 을 만들어 그 위에 마스크를 그림 → 전송량/지연 감소\n"
        "- Run은 **full inpaint 후 마스크 영역만 원본에 composite** (마스크 밖 원본 고정)\n"
    )

    st = gr.State(value=None)

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload (original)", type="filepath")
            max_side = gr.Slider(512, 2048, value=1024, step=64, label="Preview max side (속도↑: 작게, 정밀↑: 크게)")
            load_btn = gr.Button("Load to Editor", variant="secondary")

            editor = gr.ImageEditor(label="Paint mask on preview", type="filepath", height=520)

            prompt = gr.Textbox(label="Positive prompt", value="clean natural background, anime style, forest, trees, sky, sunlight")
            negative_prompt = gr.Textbox(label="Negative prompt", value="logo, watermark, text, emblem, symbol, icon, letters")

            with gr.Row():
                steps = gr.Slider(1, 60, value=28, step=1, label="Steps")
                guidance = gr.Slider(0, 20, value=5.0, step=0.5, label="CFG")
                seed = gr.Number(value=5, precision=0, label="Seed")

            with gr.Row():
                diff_thresh = gr.Slider(1, 50, value=8, step=1, label="Mask diff thresh (fallback)")
                dilate_px = gr.Slider(0, 32, value=10, step=1, label="Composite dilate(px)")
                blur_px = gr.Slider(0, 32, value=10, step=1, label="Composite blur(px)")

            with gr.Row():
                save_dir = gr.Textbox(label="Save folder", value="outputs")
                basename = gr.Textbox(label="Basename", value="sample")
            autosave = gr.Checkbox(value=True, label="Auto-save on Run")
            save_input_mask = gr.Checkbox(value=True, label="Also save input+mask")

            run_btn = gr.Button("Run Inpaint", variant="primary")

            load_info = gr.Textbox(label="Load info", interactive=False)

        with gr.Column(scale=1):
            out_img = gr.Image(label="Output", type="pil", height=520)
            mask_view = gr.Image(label="Mask(full-res, white=inpaint)", type="pil", height=520)
            info = gr.Textbox(label="Timing / Info", interactive=False)

    # 업로드 → 프리뷰 생성 → editor에 세팅
    load_btn.click(
        fn=on_upload,
        inputs=[input_img, max_side],
        outputs=[editor, st, load_info],
        concurrency_limit=1,
    )

    # Run
    run_btn.click(
        fn=run,
        inputs=[
            editor, st, prompt, negative_prompt,
            steps, guidance, seed, diff_thresh, dilate_px, blur_px,
            save_dir, basename, autosave, save_input_mask
        ],
        outputs=[out_img, mask_view, info],
        concurrency_limit=1,
    )

try:
    demo.queue(max_size=8, default_concurrency_limit=1)
except TypeError:
    demo.queue(max_size=8)

demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, share=False)
