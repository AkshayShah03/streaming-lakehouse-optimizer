"""Cost model + layout search behavioural tests."""
import numpy as np

from lakehouse.optimizer.cost_model import CostModel, encode
from lakehouse.optimizer.features import WorkloadFeatures
from lakehouse.optimizer.layout_search import recommend, candidate_layouts
from lakehouse.metrics.table_stats import Layout, DEFAULT_LAYOUT


def _wf(**kw):
    base = dict(ingest_rows_per_sec=8000, avg_selectivity=0.1,
                time_range_query_ratio=0.9, read_write_ratio=4.0,
                avg_file_mb=8.0, file_count=400, small_file_ratio=0.94,
                partition_count=16)
    base.update(kw)
    return WorkloadFeatures(**base)


def test_cost_model_recovers_signal(trained_model):
    model, (X, y_lat) = trained_model
    r2 = model.score(X, y_lat)
    assert r2 > 0.8, f"cost model failed to learn the physical signal (R^2={r2})"


def test_predict_returns_three_metrics(trained_model):
    model, _ = trained_model
    pred = model.predict(DEFAULT_LAYOUT, _wf())
    assert pred.p95_latency_ms > 0
    assert pred.storage_cost >= 0
    assert pred.write_amplification >= 0


def test_feature_vector_contract_stable():
    names = WorkloadFeatures.feature_names()
    vec = _wf().to_vector()
    assert len(names) == len(vec) == 8


def test_encode_includes_layout_and_workload():
    v = encode(DEFAULT_LAYOUT, _wf())
    # 3 layout dims + 8 workload dims
    assert len(v) == 11


def test_recommend_returns_valid_full_ranking(trained_model):
    model, _ = trained_model
    rec = recommend(model, DEFAULT_LAYOUT, _wf())
    assert len(rec.ranked) == len(candidate_layouts())
    rec.best.layout.validate()  # must be a legal layout
    # ranking is sorted ascending by objective
    objs = [s.objective for s in rec.ranked]
    assert objs == sorted(objs)


def test_small_file_workload_prefers_larger_files(trained_model):
    """A read-heavy, tiny-file table should be steered away from 64MB files."""
    model, _ = trained_model
    rec = recommend(model, Layout(64, 100, "device_bucket"),
                    _wf(avg_file_mb=6.0, small_file_ratio=0.95,
                        read_write_ratio=6.0))
    assert rec.best.layout.target_file_mb >= 128
