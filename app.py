import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image
import threading
import requests
import io
import base64

API_URL = "http://localhost:8000/api/predict"

APP_TITLE = "MedVision: Fracture Detection Client"
WINDOW_SIZE = "1050x750"

COLOR_OK = "#2CC985"
COLOR_ALERT = "#FF4444"
COLOR_WARN = "#F5A623"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


class XRayClientApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.image_path = None
        self._setup_ui()

    def _setup_ui(self):
        self.title(APP_TITLE)
        self.geometry(WINDOW_SIZE)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- САЙДБАР ---
        self.sidebar = ctk.CTkFrame(self, width=220, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(self.sidebar, text="MED VISION\nCLIENT",
                     font=ctk.CTkFont(size=20, weight="bold")).pack(pady=(30, 20))

        self.btn_load = ctk.CTkButton(self.sidebar, text="📂 Загрузить снимок",
                                      command=self.load_image)
        self.btn_load.pack(padx=20, pady=10)

        self.status_label = ctk.CTkLabel(self.sidebar, text="Готов к работе", text_color="gray")
        self.status_label.pack(side="bottom", pady=20)

        # --- ОБЛАСТЬ КОНТЕНТА ---
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.grid(row=0, column=1, padx=20, pady=20, sticky="nsew")

        # Сетка: 2 колонки (было/стало)
        self.content_frame.columnconfigure(0, weight=1)
        self.content_frame.columnconfigure(1, weight=1)

        # Исходное фото
        self.lbl_orig_title = ctk.CTkLabel(self.content_frame, text="Исходный снимок", font=("Arial", 14, "bold"))
        self.lbl_orig_title.grid(row=0, column=0, pady=5)

        self.img_label_orig = ctk.CTkLabel(self.content_frame, text="Нет изображения")
        self.img_label_orig.grid(row=1, column=0, padx=10, sticky="nsew")

        # Обработанное фото (с бэкенда)
        self.lbl_res_title = ctk.CTkLabel(self.content_frame, text="Результат AI", font=("Arial", 14, "bold"))
        self.lbl_res_title.grid(row=0, column=1, pady=5)

        self.img_label_res = ctk.CTkLabel(self.content_frame, text="Ожидание...")
        self.img_label_res.grid(row=1, column=1, padx=10, sticky="nsew")

        # --- ПАНЕЛЬ ВЕРДИКТА ---
        self.verdict_frame = ctk.CTkFrame(self.content_frame, height=100)
        self.verdict_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(20, 0))

        self.lbl_verdict = ctk.CTkLabel(self.verdict_frame, text="...", font=("Arial", 24, "bold"))
        self.lbl_verdict.pack(expand=True)

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if not path: return
        self.image_path = path

        # Показываем исходник локально сразу
        self.display_image(path, self.img_label_orig)

        # Сбрасываем правое окно
        self.img_label_res.configure(image=None, text="Отправка...")
        self.lbl_verdict.configure(text="Анализ...", text_color="white")

        # Запуск запроса в фоне
        threading.Thread(target=self.send_to_backend, daemon=True).start()

    def send_to_backend(self):
        try:
            with open(self.image_path, 'rb') as f:
                files = {'file': f}
                response = requests.post(API_URL, files=files, timeout=15)

            if response.status_code == 200:
                data = response.json()
                self.after(0, lambda: self.update_ui_success(data))
            else:
                err_msg = f"Ошибка сервера: {response.status_code}"
                self.after(0, lambda: messagebox.showerror("Error", err_msg))
                self.after(0, lambda: self.lbl_verdict.configure(text="Ошибка", text_color=COLOR_ALERT))

        except Exception as e:
            print(f"Connection error: {e}")
            self.after(0, lambda: self.lbl_verdict.configure(text="Нет соединения", text_color=COLOR_ALERT))

    def update_ui_success(self, data):
        # 1. Парсим JSON
        is_fracture = data.get("has_fracture", False)
        conf = data.get("confidence", 0.0) * 100
        img_b64_str = data.get("processed_image")

        # 2. Обновляем текст
        if is_fracture:
            text = f"🚨 ПЕРЕЛОМ ({conf:.1f}%)"
            color = COLOR_ALERT
        else:
            text = f"✅ НОРМА ({conf:.1f}%)"
            color = COLOR_OK

        self.lbl_verdict.configure(text=text, text_color=color)

        # 3. Декодируем и показываем картинку
        if img_b64_str:
            try:
                # Декодируем Base64 -> Байты
                img_bytes = base64.b64decode(img_b64_str)
                pil_img = Image.open(io.BytesIO(img_bytes))

                # Ресайзим красиво (сохраняя пропорции), чтобы влезло в UI
                # (400, 400) - максимальный размер бокса
                pil_img.thumbnail((400, 400))

                # Создаем CTkImage
                ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)

                self.img_label_res.configure(image=ctk_img, text="")
            except Exception as e:
                print(f"Ошибка декодирования картинки: {e}")
                self.img_label_res.configure(text="Ошибка картинки")
        else:
            self.img_label_res.configure(image=None, text="Нет изображения от AI")

    def display_image(self, path, label_widget):
        try:
            pil_img = Image.open(path)
            pil_img.thumbnail((400, 400))  # Такой же размер как у результата
            ctk_img = ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=pil_img.size)
            label_widget.configure(image=ctk_img, text="")
        except Exception:
            label_widget.configure(text="Ошибка файла")


if __name__ == "__main__":
    app = XRayClientApp()
    app.mainloop()