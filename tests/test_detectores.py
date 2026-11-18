from detectores import DETECTOR_FACTORIES, calcular_retardo, requires_binary_error_signal


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
