import numpy as np

from inference import OsTraceDetector


def test_postprocess_supports_binary_and_multiclass_logits():
    detector = OsTraceDetector.__new__(OsTraceDetector)
    boxes = np.array([[[0.5, 0.5, 0.2, 0.2]]], dtype=np.float32)

    for logits in (
        np.array([[[-2.0, 4.0]]], dtype=np.float32),
        np.array([[[-2.0, 3.0, 2.0]]], dtype=np.float32),
    ):
        predictions = detector.postprocess(
            [boxes, logits],
            orig_w=100,
            orig_h=100,
            conf_threshold=0.1,
        )

        assert len(predictions) == 1
        assert predictions[0]["class"] == "fracture"
        assert predictions[0]["confidence"] > 0.8


def test_postprocess_applies_nms():
    detector = OsTraceDetector.__new__(OsTraceDetector)
    boxes = np.array(
        [[[0.5, 0.5, 0.2, 0.2], [0.52, 0.52, 0.2, 0.2]]],
        dtype=np.float32,
    )
    logits = np.array([[[-2.0, 4.0], [-2.0, 3.5]]], dtype=np.float32)

    predictions = detector.postprocess(
        [boxes, logits],
        orig_w=100,
        orig_h=100,
        conf_threshold=0.1,
        iou_threshold=0.5,
    )

    assert len(predictions) == 1
    assert predictions[0]["confidence"] > 0.9
