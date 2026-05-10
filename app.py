import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import threading
import io
import base64
import numpy as np
import cv2
import os
import sys

from inference import OsTraceDetector

try:
    import pydicom

    HAS_PYDICOM = True
except ImportError:
    HAS_PYDICOM = False

APP_TITLE = "OsTrace: Standalone AI Client"
WINDOW_SIZE = "1150x750"

COLOR_OK = "#2CC985"
COLOR_ALERT = "#FF4444"
COLOR_WARN = "#F5A623"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


def encode_image_to_base64(image: np.ndarray) -> str:
    if image.dtype != np.uint8:
        image = (np.clip(image, 0, 1) * 255).astype(np.uint8)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    _, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return base64.b64encode(buffer.tobytes()).decode("utf-8")


def load_dicom(data):
    if not HAS_PYDICOM:
        raise ImportError("pydicom не установлен. Выполните: pip install pydicom")
    ds = pydicom.dcmread(io.BytesIO(data) if isinstance(data, bytes) else str(data))
    px = ds.pixel_array.astype(np.float32)
    if hasattr(ds, "WindowCenter") and hasattr(ds, "WindowWidth"):
        try:
            c = float(ds.WindowCenter[0] if isinstance(ds.WindowCenter, list) else ds.WindowCenter)
            w = float(ds.WindowWidth[0] if isinstance(ds.WindowWidth, list) else ds.WindowWidth)
            px = np.clip(px, c - w / 2, c + w / 2)
        except (ValueError, TypeError):
            pass
    mn, mx = px.min(), px.max()
    px = (px - mn) / (mx - mn) if mx > mn else np.zeros_like(px)
    if len(px.shape) == 2:
        px = np.stack([px] * 3, axis=-1)
    return px


def generate_heatmap_overlay(image: np.ndarray, predictions: list, alpha: float = 0.4) -> np.ndarray:
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    for pred in predictions:
        cx, cy = pred["x"], pred["y"]
        bw, bh = pred["width"], pred["height"]
        conf = pred["confidence"]
        sx = max(bw * 0.6, 20)
        sy = max(bh * 0.6, 20)

        y_coords, x_coords = np.mgrid[0:h, 0:w]
        gaussian = np.exp(
            -((x_coords - cx) ** 2 / (2 * sx ** 2)
              + (y_coords - cy) ** 2 / (2 * sy ** 2))
        )
        heatmap += gaussian * conf

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)

    if image.max() <= 1.0:
        vis = (image * 255).astype(np.uint8)
    else:
        vis = image.astype(np.uint8)

    if len(vis.shape) == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    elif vis.shape[2] == 3:
        vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    return cv2.addWeighted(vis, 1 - alpha, heatmap_color, alpha, 0)


# --- Основной класс приложения ---
class XRayStandaloneApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.detector = None
        self.image_refs = []

        empty_pil = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        self.empty_ctk_img = ctk.CTkImage(light_image=empty_pil, dark_image=empty_pil, size=(1, 1))

        self._setup_ui()
        self.after(100, self._auto_load_model)

    def _setup_ui(self):
        self.title(APP_TITLE)

        try:
            if sys.platform == "win32":
                self.after(200, lambda: self.iconbitmap(sys.executable))
            elif sys.platform == "darwin":
                icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "OsTrace.icns")
                if os.path.exists(icon_path):
                    self.after(200, lambda: self.iconphoto(False, ctk.CTkImage(Image.open(icon_path))))
        except Exception as e:
            print(f"Не удалось загрузить иконку: {e}")

        self.geometry(WINDOW_SIZE)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- САЙДБАР ---
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(self.sidebar, text="OSTRACE\nAI ENGINE",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(30, 20))

        # Кнопка загрузки модели (теперь ждет папку или ONNX)
        self.btn_load_model = ctk.CTkButton(self.sidebar, text="🧠 Загрузить модель",
                                            command=self.load_model, fg_color="#7b2cbf", hover_color="#5a189a")
        self.btn_load_model.pack(padx=20, pady=(0, 20))

        ctk.CTkFrame(self.sidebar, height=2, fg_color="gray30").pack(fill="x", padx=20, pady=10)

        self.btn_load_img = ctk.CTkButton(self.sidebar, text="📂 Загрузить снимки",
                                          command=self.load_images, state="disabled")
        self.btn_load_img.pack(padx=20, pady=10)

        self.use_heatmap_var = ctk.BooleanVar(value=True)
        self.cb_heatmap = ctk.CTkCheckBox(self.sidebar, text="Показывать тепловую карту",
                                          variable=self.use_heatmap_var, state="disabled")
        self.cb_heatmap.pack(padx=20, pady=10)

        self.status_label = ctk.CTkLabel(self.sidebar, text="Ожидание модели...", text_color="gray")
        self.status_label.pack(side="bottom", pady=20)

        # --- ОБЛАСТЬ КОНТЕНТА ---
        self.scroll_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

    def load_model(self):
        # Теперь ищем .onnx файлы
        path = filedialog.askopenfilename(filetypes=[("ONNX Model", "*.onnx")])
        if not path: return

        self.btn_load_model.configure(state="disabled")
        self.status_label.configure(text="Инициализация ONNX...\n(может занять время)", text_color=COLOR_WARN)

        # Новый детектор принимает директорию, в которой лежит .onnx
        model_dir = os.path.dirname(path)
        threading.Thread(target=self._init_model_backend, args=(model_dir,), daemon=True).start()

    def _auto_load_model(self):
        if getattr(sys, 'frozen', False):
            if sys.platform == "darwin" and "MacOS" in sys.executable:
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))))
            else:
                base_dir = os.path.dirname(sys.executable)
        else:
            app_dir = os.path.dirname(os.path.abspath(__file__))
            base_dirs = [app_dir, os.path.abspath(os.path.join(app_dir, ".."))]

        # Ищем папку ostracemodel (новое дефолтное место) или models
        if getattr(sys, 'frozen', False):
            base_dirs = [base_dir]

        for base_dir in base_dirs:
            for target_dir in ["ostracemodel", "models"]:
                models_dir = os.path.join(base_dir, target_dir)
                if os.path.exists(models_dir):
                    for file in os.listdir(models_dir):
                        if file.endswith(".onnx"):
                            print(f"Найдена модель по умолчанию: {models_dir}")
                            self.btn_load_model.configure(state="disabled")
                            self.status_label.configure(text="Авто-загрузка ONNX...", text_color=COLOR_WARN)
                            threading.Thread(target=self._init_model_backend, args=(models_dir,), daemon=True).start()
                            return

        print("Папка с .onnx моделью не найдена. Ждем ручной загрузки.")

    def _init_model_backend(self, model_dir):
        try:
            # Инициализация нового детектора
            self.detector = OsTraceDetector(model_dir=model_dir)
            self.after(0, self._on_model_loaded_success)
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda: self._on_model_loaded_error(err_msg))

    def _on_model_loaded_success(self):
        self.status_label.configure(text="Модель активна! ✅", text_color=COLOR_OK)
        self.btn_load_model.configure(text="Сменить модель", fg_color="gray30", state="normal")
        self.btn_load_img.configure(state="normal")
        self.cb_heatmap.configure(state="normal")

    def _on_model_loaded_error(self, err_msg):
        self.status_label.configure(text="Ошибка загрузки ❌", text_color=COLOR_ALERT)
        self.btn_load_model.configure(state="normal")
        messagebox.showerror("ML Engine Error", f"Не удалось загрузить модель:\n{err_msg}")

    def load_images(self):
        if not self.detector:
            messagebox.showwarning("Внимание", "Сначала загрузите модель!")
            return

        paths = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.png *.jpeg *.dcm")])
        if not paths: return

        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        self.image_refs.clear()

        tasks = []

        for i, path in enumerate(paths):
            row_frame = ctk.CTkFrame(self.scroll_frame, corner_radius=10)
            row_frame.pack(fill="x", pady=15, padx=10)

            filename = os.path.basename(path)
            title = ctk.CTkLabel(row_frame, text=f"Снимок {i + 1}: {filename}", font=("Arial", 16, "bold"))
            title.pack(pady=(10, 5))

            imgs_frame = ctk.CTkFrame(row_frame, fg_color="transparent")
            imgs_frame.pack(fill="x", pady=5)
            imgs_frame.columnconfigure(0, weight=1)
            imgs_frame.columnconfigure(1, weight=1)

            lbl_orig = ctk.CTkLabel(imgs_frame, text="Загрузка...", font=("Arial", 12))
            lbl_orig.grid(row=0, column=0, padx=10, pady=5)

            lbl_res = ctk.CTkLabel(imgs_frame, text="Ожидание очереди...", font=("Arial", 12))
            lbl_res.grid(row=0, column=1, padx=10, pady=5)

            lbl_verdict = ctk.CTkLabel(row_frame, text="Ожидание...", font=("Arial", 20, "bold"))
            lbl_verdict.pack(pady=(5, 15))

            self.display_image(path, lbl_orig)

            tasks.append({
                "path": path,
                "lbl_res": lbl_res,
                "lbl_verdict": lbl_verdict
            })

        threading.Thread(target=self.run_batch_inference, args=(tasks,), daemon=True).start()

    def run_batch_inference(self, tasks):
        for task in tasks:
            path = task["path"]
            lbl_res = task["lbl_res"]
            lbl_verdict = task["lbl_verdict"]

            self.after(0, lambda lr=lbl_res, lv=lbl_verdict: self._set_analyzing_state(lr, lv))

            try:
                if path.lower().endswith(('.dcm', '.dicom')):
                    image_array = load_dicom(path)
                else:
                    image_array = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

                # Вызов нового API предсказания
                result = self.detector.predict(image_array)

                heatmap_base64 = None
                if self.use_heatmap_var.get() and result["count_objects"] > 0:
                    overlay = generate_heatmap_overlay(image_array, result["predictions"])
                    heatmap_base64 = encode_image_to_base64(overlay)
                else:
                    # Если галочка снята или переломов нет, показываем оригинал (в base64 для единообразия)
                    heatmap_base64 = encode_image_to_base64(image_array)

                self.after(0, lambda r=result, hb=heatmap_base64, lr=lbl_res, lv=lbl_verdict:
                self.update_row_success(r, hb, lr, lv))

            except Exception as e:
                print(f"Error analyzing {path}: {e}")
                self.after(0, lambda lr=lbl_res, lv=lbl_verdict: self._set_error_state(lr, lv))

    def _set_analyzing_state(self, lbl_res, lbl_verdict):
        lbl_res.configure(image=self.empty_ctk_img, text="Анализ нейросетью...")
        lbl_verdict.configure(text="В процессе...", text_color="white")

    def _set_error_state(self, lbl_res, lbl_verdict):
        lbl_res.configure(image=self.empty_ctk_img, text="Ошибка обработки")
        lbl_verdict.configure(text="ОШИБКА", text_color=COLOR_ALERT)

    def update_row_success(self, result, image_base64, lbl_res, lbl_verdict):
        # Новый парсинг результатов
        count = result.get("count_objects", 0)
        is_fracture = count > 0
        preds = result.get("predictions", [])
        conf = preds[0]["confidence"] * 100 if preds else 0.0

        if is_fracture:
            text = f"🚨 НАЙДЕНО ПЕРЕЛОМОВ: {count} (Уверенность: {conf:.1f}%)"
            color = COLOR_ALERT
        else:
            text = f"✅ ПАТОЛОГИЙ НЕ НАЙДЕНО"
            color = COLOR_OK

        lbl_verdict.configure(text=text, text_color=color)

        if image_base64:
            try:
                img_bytes = base64.b64decode(image_base64)
                pil_img = Image.open(io.BytesIO(img_bytes))
                pil_img.thumbnail((400, 400))
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)

                self.image_refs.append(ctk_img)
                lbl_res.configure(image=ctk_img, text="")
            except Exception as e:
                print(f"Image display error: {e}")
                lbl_res.configure(image=self.empty_ctk_img, text="Ошибка отображения")

    def display_image(self, path, label_widget):
        try:
            if path.lower().endswith(('.dcm', '.dicom')):
                arr = load_dicom(path)
                arr_uint8 = (arr * 255).astype(np.uint8)
                pil_img = Image.fromarray(arr_uint8)
            else:
                pil_img = Image.open(path)

            pil_img.thumbnail((400, 400))
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)

            self.image_refs.append(ctk_img)
            label_widget.configure(image=ctk_img, text="")
        except Exception as e:
            print(f"Display error: {e}")
            label_widget.configure(image=self.empty_ctk_img, text="Ошибка файла")


if __name__ == "__main__":
    app = XRayStandaloneApp()
    app.mainloop()
