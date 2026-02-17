import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras
from keras import layers, applications


class SpatialAttention(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        self.conv = layers.Conv2D(
            1,
            kernel_size=7,
            padding="same",
            activation="sigmoid",
            name="attention_conv",
        )

    def call(self, x):
        avg_pool = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_pool = tf.reduce_max(x, axis=-1, keepdims=True)
        concat = tf.concat([avg_pool, max_pool], axis=-1)
        attention = self.conv(concat)
        return x * attention

    def get_config(self):
        return super().get_config()


class ChannelAttention(layers.Layer):
    def __init__(self, reduction_ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.reduction_ratio = reduction_ratio

    def build(self, input_shape):
        channels = input_shape[-1]
        self.fc1 = layers.Dense(
            channels // self.reduction_ratio,
            activation="relu",
            name="channel_fc1",
        )
        self.fc2 = layers.Dense(
            channels,
            activation="sigmoid",
            name="channel_fc2",
        )

    def call(self, x):
        gap = tf.reduce_mean(x, axis=[1, 2], keepdims=False)
        attention = self.fc1(gap)
        attention = self.fc2(attention)
        attention = tf.reshape(attention, [-1, 1, 1, tf.shape(x)[-1]])
        return x * attention

    def get_config(self):
        config = super().get_config()
        config.update({"reduction_ratio": self.reduction_ratio})
        return config


keras.saving.get_custom_objects()["SpatialAttention"] = SpatialAttention
keras.saving.get_custom_objects()["ChannelAttention"] = ChannelAttention
print(
    "[МОДЕЛЬ] Зарегистрированы SpatialAttention и ChannelAttention в custom objects"
)


def _select_backbone(backbone_name, inputs):
    if backbone_name == "efficientnetv2-b3":
        backbone = applications.EfficientNetV2B3(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
            include_preprocessing=False,
        )
    elif backbone_name == "efficientnetv2-b2":
        backbone = applications.EfficientNetV2B2(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
            include_preprocessing=False,
        )
    else:
        backbone = applications.EfficientNetV2B0(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
            include_preprocessing=False,
        )
    return backbone


def build_multitask_model(
    image_size: int = 328,
    mask_size: int = 128,
    dropout_rate: float = 0.35,
    backbone_name: str = "efficientnetv2-b3",
):
    inputs = layers.Input(shape=(image_size, image_size, 3), name="image")

    backbone = _select_backbone(backbone_name, inputs)
    backbone.trainable = True
    print(f"[МОДЕЛЬ] Мультизадачная модель, бэкбон: {backbone_name}")
    print(f"[МОДЕЛЬ] Слоёв бэкбона: {len(backbone.layers)}")

    backbone_output = backbone.output

    x_attn = ChannelAttention(reduction_ratio=16, name="channel_attention")(
        backbone_output
    )
    x_attn = SpatialAttention(name="spatial_attention")(x_attn)

    x_cls = layers.GlobalAveragePooling2D(name="gap")(x_attn)
    x_cls = layers.Dropout(dropout_rate, name="cls_dropout1")(x_cls)
    x_cls = layers.Dense(256, activation="relu", name="cls_fc1")(x_cls)
    x_cls = layers.Dropout(dropout_rate, name="cls_dropout2")(x_cls)
    classification_output = layers.Dense(
        1, activation="sigmoid", name="classification"
    )(x_cls)

    x_seg = x_attn
    x_seg = layers.Conv2DTranspose(256, 3, strides=2, padding="same", name="seg_up1")(
        x_seg
    )
    x_seg = layers.BatchNormalization(name="seg_bn1")(x_seg)
    x_seg = layers.Activation("relu", name="seg_relu1")(x_seg)
    x_seg = layers.Dropout(dropout_rate * 0.5, name="seg_drop1")(x_seg)

    x_seg = layers.Conv2DTranspose(128, 3, strides=2, padding="same", name="seg_up2")(
        x_seg
    )
    x_seg = layers.BatchNormalization(name="seg_bn2")(x_seg)
    x_seg = layers.Activation("relu", name="seg_relu2")(x_seg)
    x_seg = layers.Dropout(dropout_rate * 0.5, name="seg_drop2")(x_seg)

    x_seg = layers.Conv2DTranspose(64, 3, strides=2, padding="same", name="seg_up3")(
        x_seg
    )
    x_seg = layers.BatchNormalization(name="seg_bn3")(x_seg)
    x_seg = layers.Activation("relu", name="seg_relu3")(x_seg)
    x_seg = layers.Dropout(dropout_rate * 0.5, name="seg_drop3")(x_seg)

    x_seg = layers.Conv2DTranspose(32, 3, strides=2, padding="same", name="seg_up4")(
        x_seg
    )
    x_seg = layers.BatchNormalization(name="seg_bn4")(x_seg)
    x_seg = layers.Activation("relu", name="seg_relu4")(x_seg)

    x_seg = layers.Conv2DTranspose(16, 3, strides=2, padding="same", name="seg_up5")(
        x_seg
    )
    x_seg = layers.BatchNormalization(name="seg_bn5")(x_seg)
    x_seg = layers.Activation("relu", name="seg_relu5")(x_seg)

    current_size = image_size
    if current_size != mask_size:
        x_seg = layers.Resizing(mask_size, mask_size, name="seg_resize")(x_seg)

    segmentation_output = layers.Conv2D(
        1, 1, activation="sigmoid", name="segmentation"
    )(x_seg)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "classification": classification_output,
            "segmentation": segmentation_output,
        },
        name="multitask_fracture_model",
    )

    print(f"[МОДЕЛЬ] Выходы модели: {model.output_names}")

    return model


def build_binary_classifier(
    image_size: int = 224,
    dropout_rate: float = 0.2,
    backbone_name: str = "efficientnetv2-b3",
):
    inputs = layers.Input(shape=(image_size, image_size, 3), name="image")

    backbone = _select_backbone(backbone_name, inputs)
    backbone.trainable = True
    print(f"[МОДЕЛЬ] Бинарный классификатор, бэкбон: {backbone_name}")
    print(f"[МОДЕЛЬ] Слоёв бэкбона: {len(backbone.layers)}")

    x = backbone.output
    x = layers.GlobalAveragePooling2D(name="gap")(x)

    x = layers.Dropout(dropout_rate, name="dropout1")(x)
    x = layers.Dense(256, activation="relu", name="fc1")(x)
    x = layers.Dropout(dropout_rate, name="dropout2")(x)

    outputs = layers.Dense(1, activation="sigmoid", name="output")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model
