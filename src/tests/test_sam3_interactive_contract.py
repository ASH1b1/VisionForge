from __future__ import annotations

import numpy as np

from src.gui.models.app_state import Tool as AppTool
from src.gui.components.canvas_widget import Tool as CanvasTool
from src.gui.models.sam3_model import SAM3Model


def test_tool_sam_click_is_shared_enum():
    assert AppTool is CanvasTool
    assert AppTool.SAM_CLICK.value == "sam_click"


class _FakeProcessor:
    def __init__(self):
        self.set_image_calls = 0

    def set_image(self, image):
        self.set_image_calls += 1
        return {"image": image}


class _FakeRuntime:
    def __init__(self, *, predictor=True):
        self.inst_interactive_predictor = object() if predictor else None
        self.calls = []

    def predict_inst(self, state, **kwargs):
        self.calls.append(dict(kwargs))
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[4:12, 4:12] = 1
        logits = np.ones((1, 256, 256), dtype=np.float32)
        return np.stack([mask]), np.array([0.9], dtype=np.float32), logits


class _MultiRuntime(_FakeRuntime):
    def predict_inst(self, state, **kwargs):
        self.calls.append(dict(kwargs))
        large = np.zeros((16, 16), dtype=np.float32)
        large[2:14, 2:14] = 1
        small = np.zeros((16, 16), dtype=np.float32)
        small[6:10, 6:10] = 1
        corner = np.zeros((16, 16), dtype=np.float32)
        corner[0:3, 0:3] = 1
        masks = np.stack([large, small, corner])
        ious = np.array([0.99, 0.20, 0.50], dtype=np.float32)
        logits = np.stack(
            [
                np.full((256, 256), 1.0, dtype=np.float32),
                np.full((256, 256), 2.0, dtype=np.float32),
                np.full((256, 256), 3.0, dtype=np.float32),
            ]
        )
        return masks, ious, logits


class _Img:
    width = 16
    height = 16


def _loaded_model(*, predictor=True) -> tuple[SAM3Model, _FakeProcessor, _FakeRuntime]:
    model = SAM3Model()
    processor = _FakeProcessor()
    runtime = _FakeRuntime(predictor=predictor)
    model._processor = processor
    model._runtime = runtime
    model._loaded = True
    return model, processor, runtime


def test_predict_interactive_reuses_set_image_per_frame_key():
    model, processor, runtime = _loaded_model()
    image = _Img()
    key = (0, "frame.png", 16, 16)
    first = model.predict_interactive(
        image, box_xyxy=(1, 1, 10, 10), generation=1, frame_key=key
    )
    second = model.predict_interactive(
        image,
        point_coords=[[8.0, 8.0]],
        point_labels=[1],
        generation=2,
        frame_key=key,
    )
    assert processor.set_image_calls == 1
    assert first["generation"] == 1
    assert second["generation"] == 2
    assert len(first["polygon"]) >= 3
    assert runtime.calls[0]["multimask_output"] is False
    assert "point_coords" not in runtime.calls[0]
    assert runtime.calls[1]["normalize_coords"] is False
    assert runtime.calls[1]["multimask_output"] is False


def test_predict_interactive_without_predictor_raises_chinese():
    model, _, _ = _loaded_model(predictor=False)
    try:
        model.predict_interactive(_Img())
    except RuntimeError as exc:
        assert "interactive predictor" in str(exc)
        assert "点选" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_unload_clears_interactive_cache():
    model, processor, _ = _loaded_model()
    model.predict_interactive(_Img(), box_xyxy=(0, 0, 8, 8), frame_key="a")
    assert model._interactive_state is not None
    model.unload()
    assert model._interactive_state is None
    assert model._interactive_key is None
    assert model.has_interactive_predictor() is False
    assert processor.set_image_calls == 1


def test_clear_interactive_cache_forces_set_image():
    model, processor, _ = _loaded_model()
    key = "same"
    model.predict_interactive(_Img(), box_xyxy=(0, 0, 8, 8), frame_key=key)
    model.clear_interactive_cache()
    model.predict_interactive(_Img(), box_xyxy=(0, 0, 8, 8), frame_key=key)
    assert processor.set_image_calls == 2


def test_predict_interactive_pick_click_uses_multimask_and_smallest_hit():
    model, processor, _ = _loaded_model()
    runtime = _MultiRuntime()
    model._runtime = runtime
    result = model.predict_interactive(
        _Img(),
        point_coords=[[8.0, 8.0]],
        point_labels=[1],
        pick_click_xy=(8.0, 8.0),
        generation=3,
        frame_key="pick",
    )
    assert runtime.calls[0]["multimask_output"] is True
    assert runtime.calls[0]["normalize_coords"] is False
    assert result["click_miss"] is False
    assert len(result["polygon"]) >= 3
    xs = [p[0] for p in result["polygon"]]
    ys = [p[1] for p in result["polygon"]]
    assert max(xs) - min(xs) < 0.5
    assert max(ys) - min(ys) < 0.5
    assert float(np.asarray(result["low_res_logits"]).mean()) == 2.0
    assert processor.set_image_calls == 1


def test_predict_interactive_pick_click_miss_sets_flag():
    model, _, _ = _loaded_model()
    model._runtime = _MultiRuntime()
    result = model.predict_interactive(
        _Img(),
        point_coords=[[15.0, 15.0]],
        point_labels=[1],
        pick_click_xy=(15.0, 15.0),
        generation=4,
    )
    assert result["click_miss"] is True
    assert result["polygon"] == []
    assert result["low_res_logits"] is None


def test_predict_interactive_without_pick_click_still_single_mask():
    model, _, runtime = _loaded_model()
    model.predict_interactive(
        _Img(),
        point_coords=[[8.0, 8.0]],
        point_labels=[1],
        generation=5,
    )
    assert runtime.calls[0]["multimask_output"] is False
    assert "click_miss" not in runtime.calls[0]
