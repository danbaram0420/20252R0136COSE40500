import os
import time
import numpy as np
import gradio as gr
from PIL import Image, ImageFilter

def _to_pil_rgba(x):
    if x is None:
        return None
    if isinstance(x, Image.Image):
        return x.convert("RGBA")
    return Image.fromarray(x).convert("RGBA")

def _to_pil_rgb(x):
    if x is None:
        return None
    if isinstance(x, Image.Image):
        return x.convert("RGB")
    return Image.fromarray(x).convert("RGB")

def _mask_from_layers(layers, size_wh):
    """layers(list of PIL/np/...)의 alpha 채널로 마스크 생성 (white=inpaint)."""
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
        a = np.array(lay)[:, :, 3]  # alpha
        acc = np.maximum(acc, a)
    # alpha>0 영역을 255로
    return (acc > 0).astype(np.uint8) * 255

def _mask_from_diff(bg_rgba, comp_rgba, thresh=8):
    """fallback: background vs composite 차이로 마스크 생성."""
    bg = np.array(bg_rgba).astype(np.int16)
    cp = np.array(comp_rgba).astype(np.int16)
    diff = np.abs(cp - bg)  # (H,W,4)
    diff_mag = diff[:, :, :3].sum(axis=2) + diff[:, :, 3]
    return (diff_mag > int(thresh)).astype(np.uint8) * 255

def _maybe_dilate(mask_l, dilate_px):
    """간단 dilation(팽창). PIL MaxFilter 사용. dilate_px=0이면 그대로."""
    d = int(dilate_px)
    if d <= 0:
        return mask_l
    # MaxFilter 커널은 홀수여야 함. 대략 d 픽셀 정도 늘리려면 2d+1
    k = 2 * d + 1
    return mask_l.filter(ImageFilter.MaxFilter(size=k))

def _concat_preview(orig_rgb, mask_l):
    """(원본 | 마스크) 프리뷰 이미지 생성."""
    W, H = orig_rgb.size
    m3 = Image.merge("RGB", (mask_l, mask_l, mask_l))
    out = Image.new("RGB", (W * 2, H))
    out.paste(orig_rgb, (0, 0))
    out.paste(m3, (W, 0))
    return out

def save_mask(editor_payload, out_dir, basename, thresh, dilate_px, save_preview):
    """
    editor_payload: gr.ImageEditor output dict
    out_dir: 저장 폴더
    basename: 저장 파일 기본 이름(확장자 없이)
    """
    if editor_payload is None or not isinstance(editor_payload, dict):
        raise gr.Error("이미지를 업로드하고 마스크를 칠해 주세요.")

    bg = editor_payload.get("background", None)
    comp = editor_payload.get("composite", None)
    layers = editor_payload.get("layers", None)

    if bg is None:
        raise gr.Error("background가 없습니다. 이미지를 업로드해 주세요.")

    bg_rgba = _to_pil_rgba(bg)
    W, H = bg_rgba.size

    # 1) 가장 안정적인 방법: layers alpha
    mask_np = None
    if layers is not None:
        try:
            mask_np = _mask_from_layers(layers, (W, H))
        except Exception:
            mask_np = None

    # 2) fallback: diff
    if mask_np is None:
        if comp is None:
            raise gr.Error("composite가 없어 마스크 생성이 불가합니다. (ImageEditor 버전/설정 이슈)")
        comp_rgba = _to_pil_rgba(comp)
        if comp_rgba.size != (W, H):
            comp_rgba = comp_rgba.resize((W, H), Image.NEAREST)
        mask_np = _mask_from_diff(bg_rgba, comp_rgba, thresh=thresh)

    mask_l = Image.fromarray(mask_np).convert("L")
    mask_l = _maybe_dilate(mask_l, dilate_px)

    # 저장
    out_dir = out_dir.strip() or "masks"
    os.makedirs(out_dir, exist_ok=True)

    base = basename.strip() or "mask"
    mask_path = os.path.join(out_dir, f"{base}_mask.png")
    mask_l.save(mask_path)

    # 프리뷰 저장(옵션)
    preview_path = None
    preview_img = None
    if save_preview:
        orig_rgb = bg_rgba.convert("RGB")
        preview_img = _concat_preview(orig_rgb, mask_l)
        preview_path = os.path.join(out_dir, f"{base}_preview_orig_mask.png")
        preview_img.save(preview_path)

    info = f"Saved: {mask_path}"
    if preview_path:
        info += f"\nPreview: {preview_path}"
    info += f"\n(mask white=inpaint, size={W}x{H}, dilate={int(dilate_px)}px)"

    return mask_l, preview_img, info


with gr.Blocks(title="Mask Painter (Save mask only)") as demo:
    gr.Markdown(
        "### Mask Painter (저장 전용)\n"
        "- **추론/모델 로딩 없음** → 마스크 생성/저장 속도만 확인\n"
        "- 흰색(255) = 인페인트할 영역, 검정(0) = 유지 영역\n"
    )

    with gr.Row():
        with gr.Column(scale=1):
            editor = gr.ImageEditor(
                label="Upload image, then paint mask on top",
                type="pil",
                height=520,
            )

            out_dir = gr.Textbox(label="Output folder", value="masks")
            basename = gr.Textbox(label="Basename (no extension)", value="sample")

            with gr.Row():
                thresh = gr.Slider(1, 50, value=8, step=1,
                                   label="Diff threshold (fallback only)")
                dilate_px = gr.Slider(0, 32, value=0, step=1,
                                      label="Dilate mask (px)")

            save_preview = gr.Checkbox(value=True, label="Save preview (orig|mask)")

            save_btn = gr.Button("Save Mask", variant="primary")

        with gr.Column(scale=1):
            mask_view = gr.Image(label="Mask (white=inpaint)", type="pil", height=520)
            preview_view = gr.Image(label="Preview (orig|mask)", type="pil", height=520)
            info = gr.Textbox(label="Info", interactive=False)

    save_btn.click(
        fn=save_mask,
        inputs=[editor, out_dir, basename, thresh, dilate_px, save_preview],
        outputs=[mask_view, preview_view, info],
        concurrency_limit=1,  # Gradio 6.x 방식 (동시 클릭 방지)
    )

demo.launch(server_name="127.0.0.1", server_port=7861, show_error=True)
