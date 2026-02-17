import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
from keras import layers
from typing import Optional, Tuple, Dict, Union
import cv2


DECODER_PREFIXES = (
    "seg_up", "seg_bn", "seg_relu", "seg_drop", "seg_resize",
    "segmentation", "attention_conv", "decoder",
    "cls_", "gap", "dropout", "fc", "output", "classification",
    "channel_attention", "spatial_attention",
)


def _is_backbone_conv(layer):
    if not isinstance(layer, layers.Conv2D):
        return False
    if isinstance(layer, (layers.Conv2DTranspose, layers.DepthwiseConv2D)):
        return False
    name = layer.name.lower()
    if any(name.startswith(p) or p in name for p in DECODER_PREFIXES):
        return False
    return True


def find_best_backbone_conv(model: keras.Model) -> Optional[str]:
    for layer in model.layers:
        if layer.name == "top_conv" and _is_backbone_conv(layer):
            print(f"[GradCAM] Выбран слой: {layer.name} (top_conv)")
            return layer.name

    project_convs = []
    for layer in model.layers:
        if "project_conv" in layer.name and _is_backbone_conv(layer):
            project_convs.append(layer.name)
    if project_convs:
        chosen = project_convs[-1]
        print(f"[GradCAM] Выбран слой: {chosen} (последний project_conv)")
        return chosen

    backbone_convs = []
    for layer in model.layers:
        if _is_backbone_conv(layer) and "block" in layer.name.lower():
            backbone_convs.append(layer.name)
    if backbone_convs:
        chosen = backbone_convs[-1]
        print(f"[GradCAM] Выбран слой: {chosen} (последний block conv)")
        return chosen

    for layer in reversed(model.layers):
        if _is_backbone_conv(layer):
            print(f"[GradCAM] Выбран слой: {layer.name} (запасной вариант)")
            return layer.name

    return None


class GradCAMVisualizer:
    def __init__(
        self,
        model: keras.Model,
        conv_layer_name: Optional[str] = None,
        output_name: str = "classification",
    ):
        self.model = model
        self.output_name = output_name
        self.conv_layer_name = conv_layer_name or find_best_backbone_conv(model)
        if self.conv_layer_name is None:
            raise ValueError("[GradCAM] Свёрточный слой бэкбона не найден в модели")
        print(f"[GradCAM] Используется свёрточный слой: {self.conv_layer_name}")
        self._build_grad_model()

    def _build_grad_model(self):
        try:
            conv_layer = self.model.get_layer(self.conv_layer_name)
        except ValueError:
            self.conv_layer_name = find_best_backbone_conv(self.model)
            if self.conv_layer_name is None:
                raise ValueError("[GradCAM] Свёрточный слой бэкбона не найден")
            conv_layer = self.model.get_layer(self.conv_layer_name)

        if isinstance(self.model.output, dict):
            cls_output = self.model.output.get(
                self.output_name,
                self.model.output.get("output", list(self.model.output.values())[0]),
            )
        elif isinstance(self.model.output, (list, tuple)):
            cls_output = self.model.output[0]
        else:
            cls_output = self.model.output

        self.grad_model = keras.Model(
            inputs=self.model.inputs,
            outputs=[conv_layer.output, cls_output],
        )

    def generate_heatmap(
        self,
        image: np.ndarray,
        target_size: Tuple[int, int] = (224, 224),
    ) -> np.ndarray:
        if len(image.shape) == 3:
            image = np.expand_dims(image, 0)

        if image is None or image.size == 0:
            raise ValueError("Некорректное изображение: image равен None или пуст")

        image_tensor = tf.convert_to_tensor(image, dtype=tf.float32)

        with tf.GradientTape() as tape:
            conv_output, predictions = self.grad_model(image_tensor, training=False)

            if predictions.shape[-1] == 1:
                class_score = predictions[:, 0]
            else:
                class_score = predictions[:, 1]

        grads = tape.gradient(class_score, conv_output)

        if grads is None:
            print("[GradCAM] ПРЕДУПРЕЖДЕНИЕ: градиенты равны None — слой может быть отключён")
            return np.zeros(target_size, dtype=np.float32)

        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2)).numpy()

        conv_np = conv_output[0].numpy()
        heatmap = np.dot(conv_np, pooled_grads)

        heatmap = np.maximum(heatmap, 0)

        max_val = heatmap.max()
        if max_val > 0:
            heatmap /= max_val

        heatmap = cv2.resize(heatmap, (target_size[1], target_size[0]))

        return heatmap

    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
    ) -> np.ndarray:
        from PIL import Image as PILImage
        import matplotlib.cm as cm

        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8)

        heatmap = np.clip(heatmap, 0, 1)
        
        # Инвертируем heatmap: высокие значения = холодные цвета (синий),
        # низкие значения = тёплые цвета (красный)
        heatmap_inverted = 1.0 - heatmap

        heatmap_pil = PILImage.fromarray((heatmap_inverted * 255).astype(np.uint8))
        heatmap_resized = heatmap_pil.resize(
            (image.shape[1], image.shape[0]), PILImage.BILINEAR
        )
        heatmap_resized = np.array(heatmap_resized) / 255.0

        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        superimposed = (heatmap_colored * alpha + image * (1 - alpha)).astype(np.uint8)
        return superimposed

    def visualize(
        self,
        image: np.ndarray,
        return_heatmap: bool = False,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        h, w = image.shape[:2]
        heatmap = self.generate_heatmap(image, target_size=(h, w))
        overlay = self.overlay_heatmap(image, heatmap)

        if return_heatmap:
            return overlay, heatmap
        return overlay


def generate_gradcam_for_prediction(
    model: keras.Model,
    image: np.ndarray,
    prediction: Dict,
    image_size: int = 224,
) -> Dict:
    try:
        visualizer = GradCAMVisualizer(model)

        image_resized = cv2.resize(image, (image_size, image_size))
        if image_resized.max() > 1.0:
            image_resized = image_resized / 255.0

        overlay = visualizer.visualize(image_resized)

        overlay = cv2.resize(overlay, (image.shape[1], image.shape[0]))
        prediction["gradcam_image"] = overlay

    except Exception as e:
        print(f"[ПРЕДУПРЕЖДЕНИЕ] Генерация Grad-CAM не удалась: {e}")
        prediction["gradcam_image"] = (
            (image * 255).astype(np.uint8) if image.dtype != np.uint8 else image
        )

    return prediction
