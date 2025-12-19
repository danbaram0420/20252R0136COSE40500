import os
import sys
import time
import math
import numpy as np
from PIL import Image, ImageFilter

import torch
from diffusers import StableDiffusionXLInpaintPipeline

from PyQt6.QtCore import Qt, QRectF, QPointF, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen, QColor, QBrush
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QMessageBox,
    QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QSpinBox,
    QDoubleSpinBox, QSlider, QProgressBar, QCheckBox, QGroupBox, QFormLayout
)

# ============================================================
# 1) 네 main.py 로직과 "동일한" inpaint/composite (그대로 유지)
#    - pad_to_multiple_of_8
#    - feather_mask
#    - full_inpaint_then_composite
# ============================================================

MODEL_ID = "ShinoharaHare/Waifu-Inpaint-XL"

def pad_to_multiple_of_8(image_rgb: Image.Image, mask_l: Image.Image):
    image_rgb = image_rgb.convert("RGB")
    mask_l = mask_l.convert("L")

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

    crop_box = (0, 0, W, H)
    return image_padded, mask_padded, crop_box

def feather_mask(mask_l: Image.Image, dilate_px=10, blur_px=10):
    m = mask_l.convert("L")
    if dilate_px and dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        m = m.filter(ImageFilter.MaxFilter(size=k))
    if blur_px and blur_px > 0:
        m = m.filter(ImageFilter.GaussianBlur(radius=float(blur_px)))
    return m

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
    dilate_px: int = 10,
    blur_px: int = 10,
    progress_cb=None,
):
    image_rgb = image_rgb.convert("RGB")
    mask_l = mask_l.convert("L")

    if mask_l.size != image_rgb.size:
        mask_l = mask_l.resize(image_rgb.size, Image.NEAREST)

    img_p, msk_p, crop_box = pad_to_multiple_of_8(image_rgb, mask_l)

    gen = torch.Generator(device="cuda").manual_seed(int(seed))

    # diffusers 버전별 callback 인자 호환: 가능한 경우에만 진행률 콜백을 연결
    kwargs = dict(
        prompt=prompt,
        negative_prompt=negative_prompt if negative_prompt.strip() else None,
        image=img_p,
        mask_image=msk_p,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        width=img_p.width,
        height=img_p.height,
        generator=gen,
    )

    # callback(signature) 차이를 안전하게 처리
    if progress_cb is not None:
        def cb(step, timestep, latents):
            # step은 0..steps-1일 가능성이 큼
            try:
                progress_cb(int(step) + 1, int(steps))
            except Exception:
                pass

        try:
            kwargs["callback"] = cb
            kwargs["callback_steps"] = 1
        except Exception:
            pass

    with torch.autocast("cuda", dtype=torch.float16):
        try:
            gen_img = pipe(**kwargs).images[0].convert("RGB")
        except TypeError:
            # callback을 못 받는 경우(또는 인자 mismatch) → callback 제거하고 재시도
            kwargs.pop("callback", None)
            kwargs.pop("callback_steps", None)
            gen_img = pipe(**kwargs).images[0].convert("RGB")

    gen_img = gen_img.crop(crop_box)

    fm = feather_mask(mask_l, dilate_px=dilate_px, blur_px=blur_px)
    out = Image.composite(gen_img, image_rgb, fm)
    return out


# ============================================================
# 2) Pipeline: 전역 1회 로드 (GUI에서 재로딩 절대 금지)
# ============================================================
def load_pipe():
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

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


# ============================================================
# 3) QImage <-> PIL 변환
# ============================================================
def qimage_to_pil_rgb(qimg: QImage) -> Image.Image:
    qimg = qimg.convertToFormat(QImage.Format.Format_RGBA8888)
    w, h = qimg.width(), qimg.height()
    ptr = qimg.bits()
    ptr.setsize(h * w * 4)
    arr = np.frombuffer(ptr, np.uint8).reshape((h, w, 4))
    return Image.fromarray(arr[:, :, :3], mode="RGB")

def qimage_to_pil_l(qimg_gray: QImage) -> Image.Image:
    # grayscale8 가정
    qimg_gray = qimg_gray.convertToFormat(QImage.Format.Format_Grayscale8)
    w, h = qimg_gray.width(), qimg_gray.height()
    ptr = qimg_gray.bits()
    ptr.setsize(h * w)
    arr = np.frombuffer(ptr, np.uint8).reshape((h, w))
    return Image.fromarray(arr, mode="L")

def pil_to_qpixmap(pil_img: Image.Image) -> QPixmap:
    pil_img = pil_img.convert("RGBA")
    arr = np.array(pil_img)
    h, w, _ = arr.shape
    qimg = QImage(arr.data, w, h, 4*w, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())


# ============================================================
# 4) 캔버스: 이미지 + 마스크 레이어(흰색=인페인트)
#    - 좌클릭: 그리기
#    - 우클릭: 지우기
# ============================================================
class PaintCanvas(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setStyleSheet("background: #222;")

        self.img_q = None          # QImage RGBA
        self.mask_q = None         # QImage Grayscale8 (0/255)
        self.overlay_q = None      # QImage RGBA for visualization
        self.brush_size = 40
        self.is_painting = False
        self.is_erasing = False

    def load_image(self, path: str):
        qpix = QPixmap(path)
        if qpix.isNull():
            raise RuntimeError("이미지 로드 실패")
        self.img_q = qpix.toImage().convertToFormat(QImage.Format.Format_RGBA8888)

        w, h = self.img_q.width(), self.img_q.height()
        self.mask_q = QImage(w, h, QImage.Format.Format_Grayscale8)
        self.mask_q.fill(0)

        self.overlay_q = QImage(w, h, QImage.Format.Format_RGBA8888)
        self.overlay_q.fill(QColor(0, 0, 0, 0))

        self._refresh_view()

    def clear_mask(self):
        if self.mask_q is None:
            return
        self.mask_q.fill(0)
        self._refresh_view()

    def _img_pos_from_widget(self, pos):
        # QLabel 중앙정렬 상태에서, 표시 픽스맵 영역으로 좌표를 매핑
        pix = self.pixmap()
        if pix is None or pix.isNull():
            return None

        pw, ph = pix.width(), pix.height()
        lw, lh = self.width(), self.height()
        x0 = (lw - pw) / 2.0
        y0 = (lh - ph) / 2.0
        x = pos.x() - x0
        y = pos.y() - y0
        if x < 0 or y < 0 or x >= pw or y >= ph:
            return None

        # pixmap은 원본 크기로 표시(우리는 scale 안 함: bottleneck 최소화)
        return QPointF(x, y)

    def _paint_at(self, img_pt: QPointF, value: int):
        if self.mask_q is None:
            return
        painter = QPainter(self.mask_q)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(value, value, value)))
        r = self.brush_size / 2.0
        painter.drawEllipse(img_pt, r, r)
        painter.end()

        self._refresh_view()

    def _refresh_view(self):
        if self.img_q is None:
            self.setPixmap(QPixmap())
            return

        # overlay: 빨강 반투명으로 마스크 시각화
        w, h = self.img_q.width(), self.img_q.height()
        self.overlay_q.fill(QColor(0, 0, 0, 0))

        mask_arr = self._mask_numpy()
        # alpha: mask*0.5
        alpha = (mask_arr.astype(np.uint16) * 160 // 255).astype(np.uint8)

        rgba = np.zeros((h, w, 4), dtype=np.uint8)
        rgba[..., 0] = 255  # R
        rgba[..., 3] = alpha

        qimg_overlay = QImage(rgba.data, w, h, 4*w, QImage.Format.Format_RGBA8888).copy()
        self.overlay_q = qimg_overlay

        # compose
        base = self.img_q.copy()
        painter = QPainter(base)
        painter.drawImage(0, 0, self.overlay_q)
        painter.end()

        self.setPixmap(QPixmap.fromImage(base))

    def _mask_numpy(self):
        if self.mask_q is None:
            return None
        q = self.mask_q.convertToFormat(QImage.Format.Format_Grayscale8)
        w, h = q.width(), q.height()
        ptr = q.bits()
        ptr.setsize(h * w)
        arr = np.frombuffer(ptr, np.uint8).reshape((h, w))
        return arr

    def mousePressEvent(self, e):
        if self.img_q is None:
            return
        img_pt = self._img_pos_from_widget(e.position().toPoint())
        if img_pt is None:
            return
        if e.button() == Qt.MouseButton.LeftButton:
            self.is_painting = True
            self.is_erasing = False
            self._paint_at(img_pt, 255)
        elif e.button() == Qt.MouseButton.RightButton:
            self.is_painting = True
            self.is_erasing = True
            self._paint_at(img_pt, 0)

    def mouseMoveEvent(self, e):
        if not self.is_painting or self.img_q is None:
            return
        img_pt = self._img_pos_from_widget(e.position().toPoint())
        if img_pt is None:
            return
        self._paint_at(img_pt, 0 if self.is_erasing else 255)

    def mouseReleaseEvent(self, e):
        self.is_painting = False
        self.is_erasing = False

    def get_pil_image_and_mask(self):
        if self.img_q is None or self.mask_q is None:
            return None, None
        pil_img = qimage_to_pil_rgb(self.img_q)
        pil_msk = qimage_to_pil_l(self.mask_q)
        return pil_img, pil_msk


# ============================================================
# 5) Worker thread: 추론은 UI 스레드에서 절대 돌리지 않기
# ============================================================
class InpaintWorker(QThread):
    progress = pyqtSignal(int)   # 0..100
    finished = pyqtSignal(object, str)  # PIL.Image, info
    failed = pyqtSignal(str)

    def __init__(self, image_pil, mask_pil, params):
        super().__init__()
        self.image_pil = image_pil
        self.mask_pil = mask_pil
        self.params = params

    def run(self):
        try:
            t0 = time.perf_counter()

            def pcb(done, total):
                p = int(done * 100 / max(1, total))
                self.progress.emit(p)

            out = full_inpaint_then_composite(
                PIPE,
                self.image_pil,
                self.mask_pil,
                prompt=self.params["prompt"],
                negative_prompt=self.params["negative"],
                steps=self.params["steps"],
                guidance=self.params["guidance"],
                seed=self.params["seed"],
                dilate_px=self.params["dilate"],
                blur_px=self.params["blur"],
                progress_cb=pcb,
            )

            dt = time.perf_counter() - t0
            info = f"Done. time={dt:.2f}s | steps={self.params['steps']} cfg={self.params['guidance']} seed={self.params['seed']}"
            self.finished.emit(out, info)
        except Exception as e:
            self.failed.emit(str(e))


# ============================================================
# 6) Main window
# ============================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Waifu-Inpaint-XL (PyQt) - No Gradio Bottleneck")
        self.resize(1200, 800)

        self.canvas = PaintCanvas()
        self.out_label = QLabel("Output Preview")
        self.out_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.out_label.setStyleSheet("background:#111;color:#aaa;")

        # controls
        self.btn_open = QPushButton("Open Image")
        self.btn_clear = QPushButton("Clear Mask")
        self.btn_run = QPushButton("Run Inpaint")
        self.btn_save = QPushButton("Save Output")

        self.prompt = QLineEdit("clean natural background, anime style, forest, trees, sky, sunlight")
        self.negative = QLineEdit("logo, watermark, text, emblem, symbol, icon, letters")

        self.steps = QSpinBox(); self.steps.setRange(1, 80); self.steps.setValue(28)
        self.guidance = QDoubleSpinBox(); self.guidance.setRange(0.0, 20.0); self.guidance.setSingleStep(0.5); self.guidance.setValue(5.0)
        self.seed = QSpinBox(); self.seed.setRange(0, 2_000_000_000); self.seed.setValue(5)
        self.dilate = QSpinBox(); self.dilate.setRange(0, 64); self.dilate.setValue(10)
        self.blur = QSpinBox(); self.blur.setRange(0, 64); self.blur.setValue(10)

        self.brush = QSlider(Qt.Orientation.Horizontal); self.brush.setRange(5, 200); self.brush.setValue(40)

        self.save_dir = QLineEdit(os.path.abspath("outputs"))
        self.base = QLineEdit("sample")

        self.autosave = QCheckBox("Auto-save on finish"); self.autosave.setChecked(True)
        self.save_input_mask = QCheckBox("Also save input+mask"); self.save_input_mask.setChecked(True)

        self.prog = QProgressBar()
        self.prog.setRange(0, 100)
        self.prog.setValue(0)

        self.info = QLabel("")
        self.info.setStyleSheet("color:#ddd;")

        self.current_input = None
        self.current_mask = None
        self.current_output = None

        # layout
        left = QVBoxLayout()
        left.addWidget(self.canvas, stretch=3)

        g1 = QGroupBox("Prompts")
        f1 = QFormLayout()
        f1.addRow("Positive", self.prompt)
        f1.addRow("Negative", self.negative)
        g1.setLayout(f1)
        left.addWidget(g1)

        g2 = QGroupBox("Hyperparams")
        f2 = QFormLayout()
        f2.addRow("Steps", self.steps)
        f2.addRow("Guidance (CFG)", self.guidance)
        f2.addRow("Seed", self.seed)
        f2.addRow("Dilate(px)", self.dilate)
        f2.addRow("Blur(px)", self.blur)
        f2.addRow("Brush size", self.brush)
        g2.setLayout(f2)
        left.addWidget(g2)

        g3 = QGroupBox("Save")
        f3 = QFormLayout()
        f3.addRow("Folder", self.save_dir)
        f3.addRow("Basename", self.base)
        f3.addRow(self.autosave)
        f3.addRow(self.save_input_mask)
        g3.setLayout(f3)
        left.addWidget(g3)

        btns = QHBoxLayout()
        btns.addWidget(self.btn_open)
        btns.addWidget(self.btn_clear)
        btns.addWidget(self.btn_run)
        btns.addWidget(self.btn_save)
        left.addLayout(btns)

        left.addWidget(self.prog)
        left.addWidget(self.info)

        right = QVBoxLayout()
        right.addWidget(self.out_label, stretch=1)

        root = QHBoxLayout()
        wleft = QWidget(); wleft.setLayout(left)
        wright = QWidget(); wright.setLayout(right)
        root.addWidget(wleft, stretch=2)
        root.addWidget(wright, stretch=1)

        central = QWidget()
        central.setLayout(root)
        self.setCentralWidget(central)

        # signals
        self.btn_open.clicked.connect(self.open_image)
        self.btn_clear.clicked.connect(self.clear_mask)
        self.btn_run.clicked.connect(self.run_inpaint)
        self.btn_save.clicked.connect(self.save_output)
        self.brush.valueChanged.connect(self.on_brush_changed)

        self.worker = None

    def on_brush_changed(self, v):
        self.canvas.brush_size = int(v)

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.webp)")
        if not path:
            return
        try:
            self.canvas.load_image(path)
            self.current_output = None
            self.out_label.setPixmap(QPixmap())
            self.out_label.setText("Output Preview")
            self.info.setText(f"Loaded: {path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def clear_mask(self):
        self.canvas.clear_mask()
        self.info.setText("Mask cleared.")

    def run_inpaint(self):
        img_pil, msk_pil = self.canvas.get_pil_image_and_mask()
        if img_pil is None or msk_pil is None:
            QMessageBox.warning(self, "Warn", "먼저 이미지를 열고 마스크를 칠해 주세요.")
            return

        if (np.array(msk_pil) > 0).sum() == 0:
            QMessageBox.warning(self, "Warn", "마스크가 비어 있습니다.")
            return

        params = dict(
            prompt=self.prompt.text(),
            negative=self.negative.text(),
            steps=int(self.steps.value()),
            guidance=float(self.guidance.value()),
            seed=int(self.seed.value()),
            dilate=int(self.dilate.value()),
            blur=int(self.blur.value()),
        )

        self.current_input = img_pil
        self.current_mask = msk_pil

        self.prog.setValue(0)
        self.info.setText("Running... (GPU)")
        self.btn_run.setEnabled(False)

        self.worker = InpaintWorker(img_pil, msk_pil, params)
        self.worker.progress.connect(self.prog.setValue)
        self.worker.finished.connect(self.on_done)
        self.worker.failed.connect(self.on_fail)
        self.worker.start()

    def on_done(self, out_pil, info):
        self.current_output = out_pil
        self.out_label.setPixmap(pil_to_qpixmap(out_pil))
        self.info.setText(info)
        self.btn_run.setEnabled(True)

        if self.autosave.isChecked():
            self._save_all(auto=True)

    def on_fail(self, msg):
        QMessageBox.critical(self, "Error", msg)
        self.info.setText("Failed.")
        self.btn_run.setEnabled(True)

    def _save_all(self, auto=False):
        if self.current_output is None:
            return

        save_dir = self.save_dir.text().strip() or "outputs"
        base = self.base.text().strip() or "result"
        os.makedirs(save_dir, exist_ok=True)

        out_path = os.path.join(save_dir, f"{base}_out.png")
        self.current_output.save(out_path)

        paths = [out_path]

        if self.save_input_mask.isChecked() and self.current_input is not None and self.current_mask is not None:
            in_path = os.path.join(save_dir, f"{base}_input.png")
            mk_path = os.path.join(save_dir, f"{base}_mask.png")
            self.current_input.save(in_path)
            self.current_mask.save(mk_path)
            paths += [in_path, mk_path]

        msg = ("Auto-saved:\n" if auto else "Saved:\n") + "\n".join(paths)
        # 1) PyQt 라벨
        self.info.setText(msg)
        # 2) 콘솔 로그
        print(msg, flush=True)
        # 3) 팝업(원하면 auto일 때는 안 띄우게)
        if not auto:
            QMessageBox.information(self, "Saved", msg)

    def save_output(self):
        if self.current_output is None:
            QMessageBox.warning(self, "Warn", "저장할 결과가 없습니다. 먼저 Run을 수행하세요.")
            return
        self._save_all(auto=False)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
