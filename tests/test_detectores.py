import numpy as np
import pytest

from detectores import (
    DETECTOR_FACTORIES,
    AutoencoderDetector,
    MLPAdaptativo,
    calcular_retardo,
    es_anomalia_binaria,
    requires_binary_error_signal,
    run_detection,
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


def make_labeled_stream(n=600, drift_at=300):
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, size=(n, 3)).astype(np.float32)
    X[drift_at:] += 5.0
    y = (X[:, 0] > 0).astype(int)
    return X, y


def test_run_detection_with_labels_returns_populated_result():
    X, y = make_labeled_stream()
    result = run_detection(X, y, "ADWIN", tiene_labels=True, ground_truth=[300])
    assert result.detector == "ADWIN"
    assert 0.0 <= result.accuracy <= 1.0
    assert isinstance(result.drifts_idx, list)
    assert result.tiempo_seg >= 0.0
    assert result.ram_pico_kb >= 0.0


def test_run_detection_without_labels_still_detects_drift():
    X, _ = make_labeled_stream()
    result = run_detection(X, None, "DDM", tiene_labels=False, ground_truth=[300])
    assert len(result.drifts_idx) > 0
    assert result.accuracy == 0.0


def test_run_detection_with_labels_includes_f1_in_unit_interval():
    X, y = make_labeled_stream()
    result = run_detection(X, y, "ADWIN", tiene_labels=True, ground_truth=[300])
    assert 0.0 <= result.f1 <= 1.0


def test_run_detection_without_labels_f1_is_zero():
    X, _ = make_labeled_stream()
    result = run_detection(X, None, "DDM", tiene_labels=False, ground_truth=[300])
    assert result.f1 == 0.0


def test_es_anomalia_binaria_flags_error_far_above_frozen_baseline():
    assert es_anomalia_binaria(error=100.0, baseline_mean=1.0, baseline_std=0.1) is True


def test_es_anomalia_binaria_false_for_value_within_baseline():
    assert es_anomalia_binaria(error=1.0, baseline_mean=1.0, baseline_std=0.1) is False


def test_es_anomalia_binaria_keeps_flagging_a_sustained_shift_instead_of_adapting_to_it():
    # A diferencia de una ventana deslizante, la linea base fija no se "acostumbra"
    # a un cambio sostenido: debe seguir marcando anomalia muchas muestras despues.
    assert es_anomalia_binaria(error=10.0, baseline_mean=1.0, baseline_std=0.1) is True


def test_run_detection_sets_retardo_none_without_ground_truth():
    X, y = make_labeled_stream()
    result = run_detection(X, y, "ADWIN", tiene_labels=True, ground_truth=None)
    assert result.retardo_promedio is None


def test_run_detection_rejects_non_binary_labels_with_clear_error():
    X, y = make_labeled_stream()
    y_multiclass = y.copy()
    y_multiclass[0] = 2
    with pytest.raises(ValueError, match="binaria"):
        run_detection(X, y_multiclass, "ADWIN", tiene_labels=True, ground_truth=None)
