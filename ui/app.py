"""Streaming Lakehouse Optimizer — interactive explainer + live demo."""
import sys
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lakehouse.metrics.table_stats import SimulatedIcebergTable, Layout
from lakehouse.optimizer.train import generate_dataset
from lakehouse.optimizer.cost_model import CostModel, encode
from lakehouse.optimizer.features import WorkloadFeatures
from lakehouse.optimizer.layout_search import (
    candidate_layouts, recommend, FILE_SIZES_MB, TRIGGERS, PARTITIONS,
)
from lakehouse.maintenance.iceberg_ops import SimBackend
from lakehouse.maintenance.orchestrator import run_once
from lakehouse.maintenance.regression_guard import GuardConfig, evaluate_promotion
from lakehouse.maintenance.shadow_eval import shadow_evaluate, baseline_result
from lakehouse.maintenance.benchmark import measure_p95_latency

# ---------------------------------------------------------------------------
# Page config + CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Streaming Lakehouse Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  .block-container { padding-top: 2rem; }
  .metric-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1.2rem 1.5rem;
    text-align: center;
  }
  .metric-value-good  { font-size: 2.2rem; font-weight: 700; color: #4ade80; }
  .metric-value-bad   { font-size: 2.2rem; font-weight: 700; color: #f87171; }
  .metric-value-plain { font-size: 2.2rem; font-weight: 700; color: #e2e8f0; }
  .metric-label       { font-size: 0.82rem; color: #94a3b8; margin-top: .3rem; }
  .callout {
    border-left: 4px solid #3b82f6;
    background: #0f172a;
    padding: 0.9rem 1.1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.8rem 0;
  }
  .callout-warn {
    border-left: 4px solid #f59e0b;
    background: #0f172a;
    padding: 0.9rem 1.1rem;
    border-radius: 0 8px 8px 0;
    margin: 0.8rem 0;
  }
  .step-badge {
    display: inline-block;
    background: #1d4ed8;
    color: white;
    border-radius: 50%;
    width: 28px; height: 28px;
    text-align: center;
    line-height: 28px;
    font-weight: bold;
    margin-right: 8px;
  }
  .tag {
    display: inline-block;
    background: #1e3a5f;
    color: #93c5fd;
    border-radius: 4px;
    padding: 2px 8px;
    font-size: 0.78rem;
    margin: 2px;
  }
  h1 { font-size: 2rem !important; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached training (expensive — ~4s once, then instant on re-runs)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training cost model on 2,160 layout examples...")
def _train_model():
    X, y_lat, y_cost, y_wamp = generate_dataset(n_workloads=60, seed=1)
    rng = np.random.RandomState(1)
    idx = rng.permutation(len(X))
    cut = int(len(X) * 0.8)
    tr, te = idx[:cut], idx[cut:]
    model = CostModel().fit(X[tr], y_lat[tr], y_cost[tr], y_wamp[tr])
    r2 = model.score(X[te], y_lat[te])
    return model, r2, X, y_lat, y_cost, y_wamp


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown("## Streaming Lakehouse Optimizer")
st.markdown(
    "A real-time data pipeline that teaches itself how to stay fast — "
    "using machine learning instead of a cron job."
)
st.markdown(
    '<span class="tag">Apache Flink</span>'
    '<span class="tag">Apache Iceberg</span>'
    '<span class="tag">Apache Kafka</span>'
    '<span class="tag">Debezium CDC</span>'
    '<span class="tag">Gradient Boosting (GBM)</span>'
    '<span class="tag">Python</span>'
    '<span class="tag">Postgres</span>'
    '<span class="tag">Trino</span>',
    unsafe_allow_html=True,
)
st.divider()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs([
    "1  The Problem",
    "2  Why Machine Learning?",
    "3  Live Demo",
    "4  The Safety Net",
])


# ===========================================================================
# TAB 1 — THE PROBLEM
# ===========================================================================

with tab1:
    st.markdown("### Every 30 seconds, your pipeline creates one new file")

    c1, c2 = st.columns([1, 1], gap="large")

    with c1:
        st.markdown("""
A streaming pipeline like Flink writes data to disk at every **checkpoint** — a
safety snapshot taken every 30 seconds. Each checkpoint produces one file.

At 10,000 events per second that means:

- **2 files per minute**
- **2,880 files per day**
- each file holds only ~1–2 MB of data

The problem is that every file carries a fixed cost to open, regardless of how
much data is inside. Think of it like getting 100 envelopes each containing one
page, vs one envelope with all 100 pages. Reading 100 envelopes is slower even
if the total content is the same.
        """)
        st.markdown(
            '<div class="callout">'
            '<b>The technical term:</b> per-file open overhead — planning + metadata '
            'read cost paid once per file, before a single row of actual data is read.'
            '</div>',
            unsafe_allow_html=True,
        )

    with c2:
        # File accumulation over 48 checkpoints
        checkpoints = list(range(1, 49))
        fig = go.Figure()
        fig.add_bar(x=checkpoints, y=checkpoints, marker_color="#f87171",
                    name="Files on disk")
        fig.update_layout(
            title="Files pile up with every checkpoint flush",
            xaxis_title="Checkpoint number",
            yaxis_title="Total files on disk",
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font_color="#e2e8f0",
            margin=dict(t=40, b=40, l=40, r=20),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown("### Where query time actually goes")

    # Build a real table with 48 small files and compute cost breakdown
    @st.cache_data
    def _latency_breakdown():
        rows = []
        for n_files in [4, 8, 16, 24, 32, 48]:
            table = SimulatedIcebergTable(layout=Layout(64, 100000, "day"))
            for i in range(n_files):
                table.ingest_micro_batch(mb=1.3, rows=6500, partition=f"d{i % 4}")
            total_ms = table.scan(selectivity=0.1)
            overhead_ms = len(table._files) * table.FILE_OPEN_MS
            data_ms = total_ms - overhead_ms
            rows.append({"files": n_files, "File open overhead (ms)": overhead_ms,
                         "Actual data read (ms)": max(0, data_ms)})
        return pd.DataFrame(rows)

    df_lat = _latency_breakdown()
    fig2 = px.bar(
        df_lat, x="files",
        y=["File open overhead (ms)", "Actual data read (ms)"],
        barmode="stack",
        color_discrete_map={
            "File open overhead (ms)": "#f87171",
            "Actual data read (ms)": "#60a5fa",
        },
        labels={"files": "Number of files on disk", "value": "Latency (ms)"},
        title="As file count grows, overhead swamps actual data read time",
    )
    fig2.update_layout(
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font_color="#e2e8f0", legend_title_text="",
        margin=dict(t=40, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.markdown("### Before vs after compaction — real numbers from this project")

    m1, m2, m3, m4 = st.columns(4)
    for col, val, label, cls in [
        (m1, "48",     "Files before optimization",  "metric-value-bad"),
        (m2, "16",     "Files after optimization",   "metric-value-good"),
        (m3, "211 ms", "p95 query latency before",   "metric-value-bad"),
        (m4, "83 ms",  "p95 query latency after",    "metric-value-good"),
    ]:
        col.markdown(
            f'<div class="metric-card">'
            f'<div class="{cls}">{val}</div>'
            f'<div class="metric-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )


# ===========================================================================
# TAB 2 — WHY MACHINE LEARNING
# ===========================================================================

with tab2:
    st.markdown("### Why not just run a compaction job on a schedule?")

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown("""
A scheduled cron job compacts files at fixed intervals — say, every hour.
That works fine until anything changes:

- Ingest doubles because a new fleet of vehicles comes online
- Query patterns shift from full-table scans to narrow time-range lookups
- A different partition strategy would cut scan time in half

The cron job has no way to notice any of this. You find out months later when
someone benchmarks the table and the numbers are bad.

**What we actually need** is a function that answers: *given what this table's
queries look like right now, what file size, compaction frequency, and partition
strategy produces the fastest scans at acceptable cost?*

That answer depends on the current workload. A machine learning model can
approximate it. A cron job cannot.
        """)
        st.markdown(
            '<div class="callout-warn">'
            '<b>Why not train a model to output the optimal layout directly?</b><br>'
            'There are no ground-truth "optimal layout" labels in production — you only '
            'observe outcomes of layouts you actually ran. Instead, we train a '
            '<em>cost model</em>: a function from (layout, workload) to '
            '(predicted latency, cost, write amplification). Then we score every '
            'candidate layout and pick the best one. This is the same structure '
            'used in learned database query optimizers.'
            '</div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown("#### The 36-candidate layout search space")
        st.markdown(
            "The model scores every combination of 4 file sizes × "
            "3 compaction triggers × 3 partition strategies."
        )
        model, r2, X, y_lat, y_cost, y_wamp = _train_model()

        @st.cache_data
        def _score_all_candidates():
            wf = WorkloadFeatures(
                ingest_rows_per_sec=9000, avg_selectivity=0.07,
                time_range_query_ratio=0.9, read_write_ratio=5.0,
                avg_file_mb=1.3, file_count=48,
                small_file_ratio=0.98, partition_count=4,
            )
            rows = []
            for lay in candidate_layouts():
                pred = model.predict(lay, wf)
                rows.append({
                    "File size (MB)": lay.target_file_mb,
                    "Compaction trigger": lay.compaction_trigger_files,
                    "Partition strategy": lay.partition_granularity,
                    "Predicted p95 (ms)": round(pred.p95_latency_ms, 1),
                    "Objective score": round(pred.objective(), 2),
                })
            return pd.DataFrame(rows)

        df_cands = _score_all_candidates()
        fig3 = px.scatter(
            df_cands,
            x="File size (MB)",
            y="Predicted p95 (ms)",
            color="Partition strategy",
            size="Compaction trigger",
            hover_data=["Compaction trigger", "Objective score"],
            title="Lower = faster. Model ranks all 36 candidates instantly.",
            color_discrete_sequence=["#4ade80", "#60a5fa", "#f472b6"],
        )
        fig3.update_layout(
            plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
            font_color="#e2e8f0", margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig3, use_container_width=True)

    st.divider()
    st.markdown("### What features the model uses to make predictions")

    @st.cache_data
    def _feature_importance():
        names = [
            "target_file_mb", "compaction_trigger", "partition_granularity",
            "ingest_rows_per_sec", "avg_selectivity", "time_range_query_ratio",
            "read_write_ratio", "avg_file_mb", "file_count",
            "small_file_ratio", "partition_count",
        ]
        importances = model._latency.feature_importances_
        df = pd.DataFrame({"Feature": names, "Importance": importances})
        return df.sort_values("Importance", ascending=True)

    df_imp = _feature_importance()
    fig4 = px.bar(
        df_imp, x="Importance", y="Feature", orientation="h",
        title=f"GBM feature importances — latency head (held-out R² = {r2:.3f})",
        color="Importance",
        color_continuous_scale="blues",
    )
    fig4.update_layout(
        plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
        font_color="#e2e8f0", showlegend=False,
        margin=dict(t=40, b=40, l=160),
        coloraxis_showscale=False,
    )
    st.plotly_chart(fig4, use_container_width=True)

    st.markdown(
        '<div class="callout">'
        '<b>R² = 0.977</b> on held-out data means the model explains 97.7% of the '
        'variance in query latency across unseen workloads. The top features — '
        '<code>avg_file_mb</code> and <code>file_count</code> — directly encode '
        'the physical scan cost formula. <code>time_range_query_ratio</code> drives '
        'which partition strategy wins. 15% Gaussian measurement noise was added '
        'during training to prevent the model from simply memorizing the simulator '
        'formula (which would give R²=1.0 but generalize to nothing).'
        '</div>',
        unsafe_allow_html=True,
    )


# ===========================================================================
# TAB 3 — LIVE DEMO
# ===========================================================================

with tab3:
    st.markdown("### Run the optimizer on a simulated table")
    st.markdown(
        "Adjust the workload sliders to describe how this table is being used, "
        "then click Run. The model will score all 36 layout candidates, shadow-test "
        "the top 3 on a copy of the table, and apply the one that passes the safety gate."
    )

    model, r2, *_ = _train_model()

    c_left, c_right = st.columns([1, 1], gap="large")

    with c_left:
        st.markdown("**Workload settings**")
        ingest_rps = st.slider(
            "Ingest rate (rows/second)",
            min_value=1000, max_value=15000, value=9000, step=500,
            help="How many telemetry events the pipeline receives per second",
        )
        time_range_ratio = st.slider(
            "Time-range query ratio (0 = full scans, 1 = narrow lookups)",
            min_value=0.0, max_value=1.0, value=0.9, step=0.05,
            help="High = queries filter by time window; low = queries scan everything",
        )
        selectivity = st.slider(
            "Average query selectivity (fraction of data read)",
            min_value=0.01, max_value=0.5, value=0.07, step=0.01,
            help="0.07 = a query reads 7% of the table on average",
        )
        rw_ratio = st.slider(
            "Read/write ratio",
            min_value=0.5, max_value=10.0, value=5.0, step=0.5,
            help="5.0 = for every write, there are 5 reads",
        )

        run = st.button("Run optimizer", type="primary", use_container_width=True)

    with c_right:
        if run:
            with st.spinner("Building table and running optimizer..."):
                # Build a badly laid-out table
                prod = SimulatedIcebergTable(layout=Layout(64, 100000, "device_bucket"))
                for i in range(300):
                    prod.ingest_micro_batch(mb=0.5, rows=2500, partition=f"b{i % 16}")

                st_before = prod.stats()
                latency_before = measure_p95_latency(prod)

                wf = WorkloadFeatures(
                    ingest_rows_per_sec=float(ingest_rps),
                    avg_selectivity=float(selectivity),
                    time_range_query_ratio=float(time_range_ratio),
                    read_write_ratio=float(rw_ratio),
                    avg_file_mb=st_before.avg_file_mb,
                    file_count=st_before.file_count,
                    small_file_ratio=st_before.small_file_ratio,
                    partition_count=st_before.partition_count,
                )

                # Score all candidates
                rec = recommend(model, prod.layout, wf)
                ranked_rows = []
                for i, s in enumerate(rec.ranked[:8]):
                    shadow = shadow_evaluate(prod, s.layout)
                    ranked_rows.append({
                        "Rank": i + 1,
                        "File size": f"{s.layout.target_file_mb} MB",
                        "Trigger": s.layout.compaction_trigger_files,
                        "Partition": s.layout.partition_granularity,
                        "Predicted p95 (ms)": round(s.prediction.p95_latency_ms, 1),
                        "Shadow p95 (ms)": round(shadow.p95_latency_ms, 1),
                    })

                backend = SimBackend(prod)
                result = run_once(
                    model=model, production=prod, backend=backend, wf=wf,
                    guard=GuardConfig(min_improvement_pct=5.0),
                    ledger_path="artifacts/demo_ledger.jsonl",
                )

                st_after = prod.stats()
                latency_after = measure_p95_latency(prod)

            # Metrics
            st.markdown("**Results**")
            m1, m2 = st.columns(2)
            pct = (latency_before - latency_after) / latency_before * 100

            m1.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value-bad">{latency_before:.0f} ms</div>'
                f'<div class="metric-label">p95 query latency — before</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            cls = "metric-value-good" if latency_after < latency_before else "metric-value-bad"
            m2.markdown(
                f'<div class="metric-card">'
                f'<div class="{cls}">{latency_after:.0f} ms</div>'
                f'<div class="metric-label">p95 query latency — after ({pct:+.0f}%)</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            m3, m4 = st.columns(2)
            m3.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value-bad">{st_before.file_count}</div>'
                f'<div class="metric-label">Files before</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            m4.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value-good">{st_after.file_count}</div>'
                f'<div class="metric-label">Files after</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            promoted_color = "#4ade80" if result.promoted else "#f59e0b"
            st.markdown(
                f'<div class="callout" style="border-color: {promoted_color}">'
                f'<b>Decision:</b> {"PROMOTED" if result.promoted else "NO CHANGE"}<br>'
                f'<b>Reason:</b> {result.reason}<br>'
                f'<b>Applied layout:</b> {result.proposed}'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Candidate ranking chart
            st.markdown("**Top 8 candidates — predicted vs shadow-measured latency**")
            df_rank = pd.DataFrame(ranked_rows)
            fig5 = go.Figure()
            fig5.add_bar(
                x=df_rank["Rank"].astype(str),
                y=df_rank["Predicted p95 (ms)"],
                name="Model prediction", marker_color="#60a5fa",
            )
            fig5.add_bar(
                x=df_rank["Rank"].astype(str),
                y=df_rank["Shadow p95 (ms)"],
                name="Shadow measured", marker_color="#4ade80",
            )
            fig5.add_hline(
                y=latency_before, line_dash="dash", line_color="#f87171",
                annotation_text="Current baseline",
            )
            fig5.update_layout(
                barmode="group",
                xaxis_title="Candidate rank (1 = best predicted)",
                yaxis_title="p95 latency (ms)",
                plot_bgcolor="#0f172a", paper_bgcolor="#0f172a",
                font_color="#e2e8f0",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=40, b=40),
            )
            st.plotly_chart(fig5, use_container_width=True)

        else:
            st.info("Set the workload sliders and click **Run optimizer** to see it in action.")


# ===========================================================================
# TAB 4 — THE SAFETY NET
# ===========================================================================

with tab4:
    st.markdown("### The model can be wrong. The safety gate catches that.")
    st.markdown("""
The cost model predicts which layout is best, but predictions aren't measurements.
Before any change touches production, the proposed layout is tested on an exact
copy of the table. Only if the measured results pass three checks does the
change get applied.
    """)

    st.divider()

    # Step-by-step
    s1, s2, s3, s4 = st.columns(4)
    for col, n, title, body in [
        (s1, "1", "Clone", "Make an exact in-memory copy of the production table. The original is never touched."),
        (s2, "2", "Apply", "Run compaction on the clone using the proposed layout settings."),
        (s3, "3", "Measure", "Benchmark the clone: p95 scan latency, storage cost in dollars, write amplification in MB."),
        (s4, "4", "Guard", "All three metrics must pass their thresholds. One failure rejects the whole change."),
    ]:
        col.markdown(
            f'<div class="metric-card" style="min-height:160px">'
            f'<span class="step-badge">{n}</span><b>{title}</b><br><br>'
            f'<span style="font-size:0.85rem; color:#94a3b8">{body}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### The three guard conditions")

    g1, g2, g3 = st.columns(3)
    for col, title, body in [
        (g1, "Objective must improve by ≥5%",
         "The weighted sum of latency + cost + write amplification must be at least 5% better than the current layout. A change that is only marginally better is not worth the risk."),
        (g2, "Latency must not get worse — at all",
         "A candidate that cuts storage cost by 30% but adds 10ms to p95 latency is rejected outright. Query speed is the primary SLA. No exceptions."),
        (g3, "Write amplification within tolerance",
         "Compaction rewrites data. Rewriting 10x more than necessary burns I/O budget and can slow down live ingest. The guard enforces an absolute upper bound."),
    ]:
        col.markdown(
            f'<div class="callout" style="min-height:120px">'
            f'<b>{title}</b><br><br>'
            f'<span style="font-size:0.85rem">{body}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### See a live guard decision")

    model, *_ = _train_model()
    eg_col1, eg_col2 = st.columns(2)

    with eg_col1:
        st.markdown("**Example 1 — cheaper but slower (rejected)**")
        base_lat = st.number_input("Baseline p95 latency (ms)", value=200.0, key="b1")
        cand_lat = st.number_input("Candidate p95 latency (ms)", value=220.0, key="c1",
                                   help="Candidate is slower — should be rejected")
        base_cost = st.number_input("Baseline storage cost", value=5.0, key="bc1")
        cand_cost = st.number_input("Candidate storage cost", value=3.0, key="cc1")
        if st.button("Check this candidate", key="chk1"):
            from lakehouse.maintenance.shadow_eval import EvalResult
            b = EvalResult(layout=Layout(128, 50, "day"),
                           p95_latency_ms=base_lat, storage_cost=base_cost,
                           write_amplification=10.0, rows=1000)
            c = EvalResult(layout=Layout(256, 50, "day"),
                           p95_latency_ms=cand_lat, storage_cost=cand_cost,
                           write_amplification=10.0, rows=1000)
            d = evaluate_promotion(b, c)
            color = "#4ade80" if d.promote else "#f87171"
            verdict = "PROMOTED" if d.promote else "REJECTED"
            st.markdown(
                f'<div class="callout" style="border-color:{color}">'
                f'<b>{verdict}</b><br>{d.reason}'
                f'</div>',
                unsafe_allow_html=True,
            )

    with eg_col2:
        st.markdown("**Example 2 — faster and cheaper (promoted)**")
        base_lat2 = st.number_input("Baseline p95 latency (ms)", value=200.0, key="b2")
        cand_lat2 = st.number_input("Candidate p95 latency (ms)", value=120.0, key="c2",
                                    help="Candidate is faster — should be promoted")
        base_cost2 = st.number_input("Baseline storage cost", value=5.0, key="bc2")
        cand_cost2 = st.number_input("Candidate storage cost", value=4.5, key="cc2")
        if st.button("Check this candidate", key="chk2"):
            from lakehouse.maintenance.shadow_eval import EvalResult
            b = EvalResult(layout=Layout(128, 50, "day"),
                           p95_latency_ms=base_lat2, storage_cost=base_cost2,
                           write_amplification=10.0, rows=1000)
            c = EvalResult(layout=Layout(512, 50, "hour"),
                           p95_latency_ms=cand_lat2, storage_cost=cand_cost2,
                           write_amplification=10.0, rows=1000)
            d = evaluate_promotion(b, c)
            color = "#4ade80" if d.promote else "#f87171"
            verdict = "PROMOTED" if d.promote else "REJECTED"
            st.markdown(
                f'<div class="callout" style="border-color:{color}">'
                f'<b>{verdict}</b><br>{d.reason}'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.divider()
    st.markdown("### Every decision is logged — nothing is silent")
    st.markdown("""
Every promotion and every rejection is written to an append-only JSONL ledger
with the full input metrics, the outcome, and the reason string. The system can
never silently make the table worse. If `promotion_ledger.jsonl` exists from a
previous demo run, it appears below.
    """)

    ledger_path = Path("artifacts/promotion_ledger.jsonl")
    if ledger_path.exists():
        entries = []
        with open(ledger_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if entries:
            df_ledger = pd.DataFrame(entries)
            st.dataframe(df_ledger, use_container_width=True)
        else:
            st.info("Ledger file is empty — run the demo first.")
    else:
        st.info("No ledger yet — run the Live Demo tab first.")
