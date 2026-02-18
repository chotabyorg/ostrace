import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import threading
import io
import base64
import numpy as np
import cv2

from .gradcam import GradCAMVisualizer
from .inference import FractureDetector, Config, encode_image_to_base64, load_dicom

APP_TITLE = "OsTrace: Standalone AI Client"
WINDOW_SIZE = "1150x750"

COLOR_OK = "#2CC985"
COLOR_ALERT = "#FF4444"
COLOR_WARN = "#F5A623"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class XRayStandaloneApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self.detector = None
        self.gradcam = None

        self.current_orig_img = None
        self.current_res_img = None

        empty_pil = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        self.empty_ctk_img = ctk.CTkImage(light_image=empty_pil, dark_image=empty_pil, size=(1, 1))

        self._setup_ui()

    def _setup_ui(self):
        self.title(APP_TITLE)
        self.geometry(WINDOW_SIZE)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- САЙДБАР ---
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(self.sidebar, text="OSTRACE\nAI ENGINE",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(30, 20))

        # Кнопка загрузки модели
        self.btn_load_model = ctk.CTkButton(self.sidebar, text="🧠 Загрузить модель",
                                            command=self.load_model, fg_color="#7b2cbf", hover_color="#5a189a")
        self.btn_load_model.pack(padx=20, pady=(0, 20))

        # Разделитель
        ctk.CTkFrame(self.sidebar, height=2, fg_color="gray30").pack(fill="x", padx=20, pady=10)

        # Кнопка загрузки снимка
        self.btn_load_img = ctk.CTkButton(self.sidebar, text="📂 Загрузить снимок",
                                          command=self.load_image, state="disabled")
        self.btn_load_img.pack(padx=20, pady=10)

        # Чекбокс для Grad-CAM (Тепловая карта)
        self.use_gradcam_var = ctk.BooleanVar(value=True)
        self.cb_gradcam = ctk.CTkCheckBox(self.sidebar, text="Показывать Grad-CAM",
                                          variable=self.use_gradcam_var, state="disabled")
        self.cb_gradcam.pack(padx=20, pady=10)

        self.status_label = ctk.CTkLabel(self.sidebar, text="Ожидание модели...", text_color="gray")
        self.status_label.pack(side="bottom", pady=20)

        # --- ОБЛАСТЬ КОНТЕНТА ---
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.columnconfigure(1, weight=1)

        # Исходное фото
        self.lbl_orig_title = ctk.CTkLabel(self.content_frame, text="Исходный снимок", font=("Arial", 14, "bold"))
        self.lbl_orig_title.grid(row=0, column=0, pady=5)
        self.img_label_orig = ctk.CTkLabel(self.content_frame, text="Нет изображения")
        self.img_label_orig.grid(row=1, column=0, padx=10, sticky="nsew")

        # Обработанное фото
        self.lbl_res_title = ctk.CTkLabel(self.content_frame, text="Результат AI (Сегментация / Внимание)",
                                          font=("Arial", 14, "bold"))
        self.lbl_res_title.grid(row=0, column=1, pady=5)
        self.img_label_res = ctk.CTkLabel(self.content_frame, text="Ожидание...")
        self.img_label_res.grid(row=1, column=1, padx=10, sticky="nsew")

        # --- ПАНЕЛЬ ВЕРДИКТА ---
        self.verdict_frame = ctk.CTkFrame(self.content_frame, height=100)
        self.verdict_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(20, 0))

        self.lbl_verdict = ctk.CTkLabel(self.verdict_frame, text="...", font=("Arial", 24, "bold"))
        self.lbl_verdict.pack(expand=True)

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("Keras Model", "*.keras *.h5")])
        if not path: return

        self.btn_load_model.configure(state="disabled")
        self.status_label.configure(text="Инициализация ML-движка...\n(может занять 10-15 сек)", text_color=COLOR_WARN)

        # Загружаем тяжелую модель в фоне, чтобы не вешать UI
        threading.Thread(target=self._init_model_backend, args=(path,), daemon=True).start()

    def _init_model_backend(self, model_path):
        try:
            # Инициализация детекторов из кода ML-инженеров
            config = Config()
            self.detector = FractureDetector(config=config, model_path=model_path)

            # Автоматически обновляем размер картинки под требования загруженной модели
            if self.detector.model is not None and self.detector.model.input_shape:
                expected_size = self.detector.model.input_shape[1]
                if expected_size:
                    self.detector.config.image_size = expected_size
                    print(f"Размер в конфиге автоматически изменен на {expected_size}")

            try:
                self.gradcam = GradCAMVisualizer(self.detector.model)
            except Exception as e:
                print(f"GradCAM init warning: {e}")
                self.gradcam = None

            self.after(0, self._on_model_loaded_success)
        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda: self._on_model_loaded_error(err_msg))

    def _on_model_loaded_success(self):
        self.status_label.configure(text="Модель активна! ✅", text_color=COLOR_OK)
        self.btn_load_model.configure(text="Модель загружена", fg_color="gray30", state="normal")
        self.btn_load_img.configure(state="normal")
        if self.gradcam:
            self.cb_gradcam.configure(state="normal")

    def _on_model_loaded_error(self, err_msg):
        self.status_label.configure(text="Ошибка загрузки модели ❌", text_color=COLOR_ALERT)
        self.btn_load_model.configure(state="normal")
        messagebox.showerror("ML Engine Error", f"Не удалось загрузить модель:\n{err_msg}")

    def load_image(self):
        if not self.detector:
            messagebox.showwarning("Внимание", "Сначала загрузите модель!")
            return

        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg *.dcm")])
        if not path: return
        self.image_path = path

        self.display_image(path, self.img_label_orig)
        self.img_label_res.configure(image=self.empty_ctk_img, text="Нейросеть анализирует...")
        self.lbl_verdict.configure(text="Анализ...", text_color="white")

        threading.Thread(target=self.run_inference, daemon=True).start()

    def run_inference(self):
        try:
            # 1. Загрузка изображения в виде массива
            if self.image_path.lower().endswith(('.dcm', '.dicom')):
                image_array = load_dicom(self.image_path)
            else:
                image_array = np.array(Image.open(self.image_path).convert("RGB"), dtype=np.float32) / 255.0

            # 2. Прямой вызов модели (передаем готовый массив, а не путь)
            result = self.detector.predict(image_array, return_visualization=True)

            # 3. Опционально: генерация Grad-CAM
            gradcam_base64 = None
            if self.use_gradcam_var.get() and self.gradcam is not None:
                image_size = self.detector.config.image_size
                image_resized = cv2.resize(image_array, (image_size, image_size))

                overlay = self.gradcam.visualize(image_resized)
                original_h, original_w = image_array.shape[:2]
                overlay = cv2.resize(overlay, (original_w, original_h))
                gradcam_base64 = encode_image_to_base64(overlay)

            # Передаем результаты в UI поток
            self.after(0, lambda: self.update_ui_success(result, gradcam_base64))

        except Exception as e:
            print(f"Inference error: {e}")
            self.after(0, lambda: self.lbl_verdict.configure(text="Ошибка анализа", text_color=COLOR_ALERT))

    def update_ui_success(self, result, gradcam_base64):
        is_fracture = result.get("has_fracture", False)
        conf = result.get("confidence", 0.0) * 100

        img_b64_str = gradcam_base64 if gradcam_base64 else result.get("processed_image")

        if is_fracture:
            text = f"🚨 ПЕРЕЛОМ ОБНАРУЖЕН ({conf:.1f}%)"
            color = COLOR_ALERT
        else:
            text = f"✅ ПАТОЛОГИЙ НЕ НАЙДЕНО ({conf:.1f}%)"
            color = COLOR_OK

        self.lbl_verdict.configure(text=text, text_color=color)

        if img_b64_str:
            try:
                img_bytes = base64.b64decode(img_b64_str)
                pil_img = Image.open(io.BytesIO(img_bytes))
                pil_img.thumbnail((450, 450))
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
                self.current_res_img = ctk_img
                self.img_label_res.configure(image=self.current_res_img, text="")
            except Exception as e:
                print(f"Image display error: {e}")
                self.img_label_res.configure(image=self.empty_ctk_img, text="Ошибка отображения")

    def display_image(self, path, label_widget):
        try:
            if path.lower().endswith(('.dcm', '.dicom')):
                arr = load_dicom(path)
                arr_uint8 = (arr * 255).astype(np.uint8)
                pil_img = Image.fromarray(arr_uint8)
            else:
                pil_img = Image.open(path)

            pil_img.thumbnail((450, 450))
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            if label_widget == self.img_label_orig:
                self.current_orig_img = ctk_img
                label_widget.configure(image=self.current_orig_img, text="")
            else:
                self.current_res_img = ctk_img
                label_widget.configure(image=self.current_res_img, text="")

        except Exception as e:
            print(f"Display error: {e}")
            label_widget.configure(image=self.empty_ctk_img, text="Ошибка файла")


if __name__ == "__main__":
    app = XRayStandaloneApp()
    app.mainloop()