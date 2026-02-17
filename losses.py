import os

os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import keras


class WarmupSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, base_schedule=None, initial_lr=0.001):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.base_schedule = base_schedule
        self.initial_lr = initial_lr

    def __call__(self, step):
        def warmup():
            return self.initial_lr * tf.cast(step, tf.float32) / tf.cast(
                self.warmup_steps, tf.float32
            )

        def after_warmup():
            if self.base_schedule is not None:
                return self.base_schedule(step - self.warmup_steps)
            return self.initial_lr

        return tf.cond(step < self.warmup_steps, warmup, after_warmup)

    def get_config(self):
        return {
            "warmup_steps": self.warmup_steps,
            "initial_lr": self.initial_lr,
        }

    @classmethod
    def from_config(cls, config):
        return cls(
            warmup_steps=config["warmup_steps"],
            initial_lr=config.get("initial_lr", 0.001),
        )


def combined_segmentation_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

    per_sample_mask_sum = tf.reduce_sum(y_true, axis=[1, 2, 3])

    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    dice = (2.0 * intersection + 1e-6) / (
        tf.reduce_sum(y_true, axis=[1, 2, 3])
        + tf.reduce_sum(y_pred, axis=[1, 2, 3])
        + 1e-6
    )
    dice_loss_per_sample = 1 - dice

    bce = keras.losses.binary_crossentropy(y_true, y_pred)
    bce_loss_per_sample = tf.reduce_mean(bce, axis=[1, 2])

    loss_per_sample = dice_loss_per_sample + bce_loss_per_sample

    mask = tf.cast(per_sample_mask_sum > 0, tf.float32)

    masked_loss = loss_per_sample * mask

    num_positive = tf.reduce_sum(mask)
    return tf.reduce_sum(masked_loss) / (num_positive + 1e-6)


def iou_metric(y_true, y_pred, threshold=0.5, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred > threshold, tf.float32)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection

    return (intersection + smooth) / (union + smooth)


def focal_loss(gamma=2.0, alpha=0.75):
    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
        focal_weight = alpha_t * tf.pow(1 - p_t, gamma)
        bce = -tf.math.log(p_t)
        return tf.reduce_mean(focal_weight * bce)

    return loss


def weighted_bce(pos_weight):
    pos_weight = min(pos_weight, 3.0)

    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        bce = -(
            y_true * tf.math.log(y_pred) + (1 - y_true) * tf.math.log(1 - y_pred)
        )

        weights = y_true * pos_weight + (1 - y_true) * 1.0

        return tf.reduce_mean(bce * weights)

    return loss


def binary_crossentropy_with_label_smoothing(smoothing=0.1):
    def loss(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        y_true_smooth = y_true * (1 - smoothing) + 0.5 * smoothing

        y_pred = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)

        bce = -(
            y_true_smooth * tf.math.log(y_pred)
            + (1 - y_true_smooth) * tf.math.log(1 - y_pred)
        )

        return tf.reduce_mean(bce)

    return loss


keras.saving.get_custom_objects()["WarmupSchedule"] = WarmupSchedule
keras.saving.get_custom_objects()["combined_segmentation_loss"] = combined_segmentation_loss
keras.saving.get_custom_objects()["iou_metric"] = iou_metric
