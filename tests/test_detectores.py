import numpy as np

from detectores import (
    DETECTOR_FACTORIES,
    AutoencoderDetector,
    MLPAdaptativo,
    calcular_retardo,
    requires_binary_error_signal,
)


def test_all_seven_detectors_are_registered():
    assert set(DETECTOR_FACTORIES) == {
        "ADWIN",
        "DDM",
        "EDDM",
        "HDDM_A",
        "HDDM_W",
        "PageHinkley",
        "KSWIN",
    }


def test_factories_return_fresh_instances_each_call():
    a = DETECTOR_FACTORIES["ADWIN"]()
    b = DETECTOR_FACTORIES["ADWIN"]()
    assert a is not b


def test_requires_binary_error_signal_partitions_detectors():
    assert requires_binary_error_signal("DDM") is True
    assert requires_binary_error_signal("ADWIN") is False


def test_calcular_retardo_averages_nearest_following_detections():
    assert calcular_retardo(drifts_idx=[5010, 9050], ground_truth=[5000, 9000], horizonte=2000) == 30.0


def test_calcular_retardo_returns_none_without_ground_truth():
    assert calcular_retardo(drifts_idx=[100], ground_truth=None) is None


def test_mlp_predict_proba_one_returns_value_in_unit_interval():
    mlp = MLPAdaptativo(input_dim=4)
    p = mlp.predict_proba_one(np.random.rand(4).astype(np.float32))
    assert 0.0 <= p <= 1.0


def test_mlp_learn_one_changes_predictions_over_many_updates():
    mlp = MLPAdaptativo(input_dim=2)
    x = np.array([1.0, 1.0], dtype=np.float32)
    before = mlp.predict_proba_one(x)
    for _ in range(200):
        mlp.learn_one(x, 1)
    after = mlp.predict_proba_one(x)
    assert after > before


def test_mlp_reset_reinitializes_weights():
    mlp = MLPAdaptativo(input_dim=2)
    x = np.array([1.0, 1.0], dtype=np.float32)
    for _ in range(200):
        mlp.learn_one(x, 1)
    trained = mlp.predict_proba_one(x)
    mlp.reset()
    reset_val = mlp.predict_proba_one(x)
    assert reset_val != trained


def test_warmup_handles_fewer_rows_than_default_window():
    ae = AutoencoderDetector(input_dim=3)
    ae.warmup(np.random.rand(10, 3).astype(np.float32))


def test_reconstruction_error_is_non_negative_float():
    ae = AutoencoderDetector(input_dim=3)
    ae.warmup(np.random.rand(50, 3).astype(np.float32))
    err = ae.reconstruction_error(np.random.rand(3).astype(np.float32))
    assert err >= 0.0


def test_is_drift_flags_large_sustained_error_jump():
    ae = AutoencoderDetector(input_dim=2)
    normal = np.tile(np.array([0.5, 0.5], dtype=np.float32), (100, 1))
    ae.warmup(normal)
    flagged = False
    for _ in range(50):
        err = ae.reconstruction_error(np.array([0.5, 0.5], dtype=np.float32))
        flagged = flagged or ae.is_drift(err)
    for _ in range(50):
        err = ae.reconstruction_error(np.array([50.0, -50.0], dtype=np.float32))
        flagged = flagged or ae.is_drift(err)
    assert flagged
