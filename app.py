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
        self.detector = None
        self.gradcam = None

        self.image_refs = []

        empty_pil = Image.new("RGBA", (1, 1), (0, 0, 0, 0))
        self.empty_ctk_img = ctk.CTkImage(light_image=empty_pil, dark_image=empty_pil, size=(1, 1))

        self._setup_ui()
        self.after(100, self._auto_load_model)

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

        # Кнопка загрузки снимков
        self.btn_load_img = ctk.CTkButton(self.sidebar, text="📂 Загрузить снимки",
                                          command=self.load_images, state="disabled")
        self.btn_load_img.pack(padx=20, pady=10)

        # Чекбокс для Grad-CAM (Тепловая карта)
        self.use_gradcam_var = ctk.BooleanVar(value=True)
        self.cb_gradcam = ctk.CTkCheckBox(self.sidebar, text="Показывать Grad-CAM",
                                          variable=self.use_gradcam_var, state="disabled")
        self.cb_gradcam.pack(padx=20, pady=10)

        self.status_label = ctk.CTkLabel(self.sidebar, text="Ожидание модели...", text_color="gray")
        self.status_label.pack(side="bottom", pady=20)

        # --- ОБЛАСТЬ КОНТЕНТА ---
        self.scroll_frame = ctk.CTkScrollableFrame(self, fg_color="transparent")
        self.scroll_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

    def load_model(self):
        path = filedialog.askopenfilename(filetypes=[("Keras Model", "*.keras *.h5")])
        if not path: return

        self.btn_load_model.configure(state="disabled")
        self.status_label.configure(text="Инициализация ML-движка...\n(может занять 10-15 сек)", text_color=COLOR_WARN)

        # Загружаем тяжелую модель в фоне, чтобы не вешать UI
        threading.Thread(target=self._init_model_backend, args=(path,), daemon=True).start()

    def _auto_load_model(self):
        # 1. Определяем, где лежит наша программа
        if getattr(sys, 'frozen', False):
            if sys.platform == "darwin" and "MacOS" in sys.executable:
                # Если это Mac .app файл: спускаемся на 4 уровня вверх из папки Contents/MacOS/
                base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(sys.executable))))
            else:
                base_dir = os.path.dirname(sys.executable)
        else:
            # Если запускаем из PyCharm
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

        # 2. Ищем папку models рядом с программой
        models_dir = os.path.join(base_dir, "models")

        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith((".keras", ".h5")):
                    model_path = os.path.join(models_dir, file)
                    print(f"Найдена модель по умолчанию: {model_path}")

                    self.btn_load_model.configure(state="disabled")
                    self.status_label.configure(text="Авто-загрузка модели...\n(может занять 10-15 сек)",
                                                text_color=COLOR_WARN)

                    threading.Thread(target=self._init_model_backend, args=(model_path,), daemon=True).start()
                    return

        print("Папка models не найдена или пуста. Ждем ручной загрузки.")

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
        self.btn_load_model.configure(text="Сменить модель", fg_color="gray30", state="normal")
        self.btn_load_img.configure(state="normal")
        if self.gradcam:
            self.cb_gradcam.configure(state="normal")

    def _on_model_loaded_error(self, err_msg):
        self.status_label.configure(text="Ошибка загрузки модели ❌", text_color=COLOR_ALERT)
        self.btn_load_model.configure(state="normal")
        messagebox.showerror("ML Engine Error", f"Не удалось загрузить модель:\n{err_msg}")

    def load_images(self):
        if not self.detector:
            messagebox.showwarning("Внимание", "Сначала загрузите модель!")
            return

        paths = filedialog.askopenfilenames(filetypes=[("Images", "*.jpg *.png *.jpeg *.dcm")])
        if not paths: return

        # Очищаем ленту от предыдущих результатов
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        self.image_refs.clear()

        tasks = []

        # Динамически создаем UI-карточку для каждого снимка
        for i, path in enumerate(paths):
            row_frame = ctk.CTkFrame(self.scroll_frame, corner_radius=10)
            row_frame.pack(fill="x", pady=15, padx=10)

            # Заголовок карточки с именем файла
            filename = path.split("/")[-1]
            title = ctk.CTkLabel(row_frame, text=f"Снимок {i + 1}: {filename}", font=("Arial", 16, "bold"))
            title.pack(pady=(10, 5))

            # Контейнер для двух картинок
            imgs_frame = ctk.CTkFrame(row_frame, fg_color="transparent")
            imgs_frame.pack(fill="x", pady=5)
            imgs_frame.columnconfigure(0, weight=1)
            imgs_frame.columnconfigure(1, weight=1)

            lbl_orig = ctk.CTkLabel(imgs_frame, text="Загрузка...", font=("Arial", 12))
            lbl_orig.grid(row=0, column=0, padx=10, pady=5)

            lbl_res = ctk.CTkLabel(imgs_frame, text="Ожидание очереди...", font=("Arial", 12))
            lbl_res.grid(row=0, column=1, padx=10, pady=5)

            # Вердикт внизу карточки
            lbl_verdict = ctk.CTkLabel(row_frame, text="Ожидание...", font=("Arial", 20, "bold"))
            lbl_verdict.pack(pady=(5, 15))

            # Сразу показываем оригинальную картинку, чтобы интерфейс не казался зависшим
            self.display_image(path, lbl_orig)

            # Сохраняем ссылки на элементы UI, чтобы передать их в поток анализа
            tasks.append({
                "path": path,
                "lbl_res": lbl_res,
                "lbl_verdict": lbl_verdict
            })

        # Запускаем пакетную обработку в отдельном потоке
        threading.Thread(target=self.run_batch_inference, args=(tasks,), daemon=True).start()

    def run_batch_inference(self, tasks):
        # Обрабатываем снимки по очереди
        for task in tasks:
            path = task["path"]
            lbl_res = task["lbl_res"]
            lbl_verdict = task["lbl_verdict"]

            # Обновляем UI: показываем, что именно этот снимок сейчас в работе
            self.after(0, lambda lr=lbl_res, lv=lbl_verdict: self._set_analyzing_state(lr, lv))

            try:
                if path.lower().endswith(('.dcm', '.dicom')):
                    image_array = load_dicom(path)
                else:
                    image_array = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

                result = self.detector.predict(image_array, return_visualization=True)

                gradcam_base64 = None
                if self.use_gradcam_var.get() and self.gradcam is not None:
                    image_size = self.detector.config.image_size
                    image_resized = cv2.resize(image_array, (image_size, image_size))
                    overlay = self.gradcam.visualize(image_resized)
                    original_h, original_w = image_array.shape[:2]
                    overlay = cv2.resize(overlay, (original_w, original_h))
                    gradcam_base64 = encode_image_to_base64(overlay)

                # Используем default arguments в lambda, чтобы зафиксировать значения переменных в цикле
                self.after(0, lambda r=result, g=gradcam_base64, lr=lbl_res, lv=lbl_verdict:
                self.update_row_success(r, g, lr, lv))

            except Exception as e:
                print(f"Error analyzing {path}: {e}")
                self.after(0, lambda lr=lbl_res, lv=lbl_verdict:
                self._set_error_state(lr, lv))

    def _set_analyzing_state(self, lbl_res, lbl_verdict):
        lbl_res.configure(image=self.empty_ctk_img, text="Анализ нейросетью...")
        lbl_verdict.configure(text="В процессе...", text_color="white")

    def _set_error_state(self, lbl_res, lbl_verdict):
        lbl_res.configure(image=self.empty_ctk_img, text="Ошибка обработки")
        lbl_verdict.configure(text="ОШИБКА", text_color=COLOR_ALERT)

    def update_row_success(self, result, gradcam_base64, lbl_res, lbl_verdict):
        is_fracture = result.get("has_fracture", False)
        conf = result.get("confidence", 0.0) * 100

        img_b64_str = gradcam_base64 if gradcam_base64 else result.get("processed_image")

        if is_fracture:
            text = f"🚨 ПЕРЕЛОМ ОБНАРУЖЕН ({conf:.1f}%)"
            color = COLOR_ALERT
        else:
            text = f"✅ ПАТОЛОГИЙ НЕ НАЙДЕНО ({conf:.1f}%)"
            color = COLOR_OK

        lbl_verdict.configure(text=text, text_color=color)

        if img_b64_str:
            try:
                img_bytes = base64.b64decode(img_b64_str)
                pil_img = Image.open(io.BytesIO(img_bytes))
                # Чуть уменьшил размер до 400, чтобы удобнее скроллилось
                pil_img.thumbnail((400, 400))
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)

                self.image_refs.append(ctk_img)  # Спасаем от сборщика мусора
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