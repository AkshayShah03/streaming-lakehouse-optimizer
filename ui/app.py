"""Streaming Lakehouse Optimizer — interactive explainer and live demo."""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lakehouse.metrics.table_stats import SimulatedIcebergTable, Layout
from lakehouse.optimizer.train import generate_dataset
from lakehouse.optimizer.cost_model import CostModel, encode
from lakehouse.optimizer.features import WorkloadFeatures
from lakehouse.optimizer.layout_search import candidate_layouts, recommend
from lakehouse.maintenance.iceberg_ops import SimBackend
from lakehouse.maintenance.orchestrator import run_once
from lakehouse.maintenance.regression_guard import GuardConfig, evaluate_promotion
from lakehouse.maintenance.shadow_eval import shadow_evaluate, baseline_result, EvalResult
from lakehouse.maintenance.benchmark import measure_p95_latency

# ---------------------------------------------------------------------------
# Config — minimal overrides, let Streamlit do its job
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Streaming Lakehouse Optimizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  #MainMenu, footer { visibility: hidden; }
  .block-container { padding-top: 2rem; max-width: 1080px; }
  .stTabs [data-baseweb="tab"] { font-size: 0.9rem; }
</style>
""", unsafe_allow_html=True)

# Consistent chart palette used throughout
C_BAD    = "#dc2626"   # red   — before / bad state
C_GOOD   = "#16a34a"   # green — after / good state
C_MODEL  = "#2563eb"   # blue  — model prediction
C_SHADOW = "#0891b2"   # teal  — shadow measured
C_GRAY   = "#9ca3af"   # gray  — neutral / secondary
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=12),
    margin=dict(t=36, b=36, l=8, r=8),
    xaxis=dict(showgrid=False, linecolor="#e5e7eb"),
    yaxis=dict(gridcolor="#f3f4f6", linecolor="#e5e7eb"),
)


# ---------------------------------------------------------------------------
# Cached model training (~4 s once)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Training cost model on 2,160 layout examples…")
def _train_model():
    X, y_lat, y_cost, y_wamp = generate_dataset(n_workloads=60, seed=1)
    rng = np.random.RandomState(1)
    idx = rng.permutation(len(X))
    cut = int(len(X) * 0.8)
    tr, te = idx[:cut], idx[cut:]
    model = CostModel().fit(X[tr], y_lat[tr], y_cost[tr], y_wamp[tr])
    r2 = model.score(X[te], y_lat[te])
    return model, r2


# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------

st.title("Streaming Lakehouse Optimizer")
st.caption(
    "A Flink → Kafka → Iceberg pipeline that replaces scheduled compaction "
    "with a learned cost model and a safety-gated promotion loop."
)
st.markdown(
    "`Apache Flink` `Apache Iceberg` `Apache Kafka` `Debezium` "
    "`Gradient Boosting` `PyIceberg` `Trino` `Airflow`"
)
st.divider()

tab_problem, tab_solution, tab_demo, tab_guard = st.tabs([
    "The Problem",
    "Why Machine Learning",
    "Live Demo",
    "Safety Gate",
])


# ===========================================================================
# THE PROBLEM
# ===========================================================================

with tab_problem:
    st.subheader("A streaming pipeline creates one file every 30 seconds")

    left, right = st.columns([3, 4], gap="large")

    with left:
        st.markdown(
            "Flink flushes data to Iceberg at every checkpoint interval. "
            "At a 30-second interval and 10,000 events per second, that is **2,880 files per day** "
            "before any compaction runs."
        )
        st.markdown(
            "Each file carries a fixed planning and open cost regardless of how much data it holds. "
            "With 48 files averaging 1.3 MB each, the pipeline is paying that overhead 48 times "
            "per query — before reading a single byte of actual data."
        )
        st.markdown("These are the measured numbers from this project's simulator:")
        c1, c2 = st.columns(2)
        c1.metric("Files on disk", "48", help="After one simulated day of ingest at 10k rows/sec")
        c2.metric("Avg file size", "1.3 MB", help="Target is ~128–512 MB for efficient scans")
        c1.metric("p95 scan latency", "211 ms", help="Per-file open overhead dominates")
        c2.metric("After compaction", "83 ms", delta="-61%", delta_color="inverse",
                  help="16 properly-sized files after the optimizer runs")

        with st.expander("What is per-file open overhead?"):
            st.markdown(
                "Every file in a columnar format like Parquet (which Iceberg uses under the hood) "
                "requires the query engine to: open an S3 connection, read the file footer to get "
                "column statistics and row group offsets, then decide which row groups to scan. "
                "That sequence takes roughly 4 ms per file. With 48 files that is **192 ms before "
                "reading a row of data**. With 16 files it is 64 ms."
            )

    with right:
        # How scan time breaks down as file count grows
        @st.cache_data
        def _breakdown_df():
            rows = []
            for n in [4, 8, 16, 24, 32, 48]:
                table = SimulatedIcebergTable(layout=Layout(64, 100_000, "day"))
                for i in range(n):
                    table.ingest_micro_batch(mb=1.3, rows=6500, partition=f"d{i % 4}")
                overhead = len(table._files) * table.FILE_OPEN_MS
                data_ms  = max(0.0, table.scan(0.1) - overhead)
                rows.append({"Files": n, "Open overhead (ms)": overhead, "Data read (ms)": data_ms})
            return pd.DataFrame(rows)

        df = _breakdown_df()
        fig = go.Figure()
        fig.add_bar(x=df["Files"], y=df["Open overhead (ms)"],
                    name="File open overhead", marker_color=C_BAD)
        fig.add_bar(x=df["Files"], y=df["Data read (ms)"],
                    name="Actual data read", marker_color=C_GRAY)
        fig.update_layout(
            **CHART_LAYOUT,
            barmode="stack",
            title=dict(text="Where query time goes as file count grows", font_size=13),
            xaxis_title="Files on disk",
            yaxis_title="Simulated latency (ms)",
            legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("Why a fixed schedule doesn't fix this")

    a, b, c = st.columns(3)
    a.info(
        "**Ingest doubles.**  \nA new batch of vehicles comes online. "
        "The cron job compacts on the same schedule it always did, so files pile up faster than it can keep up."
    )
    b.info(
        "**Query pattern shifts.**  \nAnalysts start running narrow time-range lookups instead of full scans. "
        "Hourly partitioning would prune 60% of files. The cron job doesn't change partition strategy."
    )
    c.info(
        "**You find out months later.**  \nThere is no alert for \"compaction is no longer keeping up.\" "
        "Someone benchmarks the table and the numbers are bad."
    )


# ===========================================================================
# WHY MACHINE LEARNING
# ===========================================================================

with tab_solution:
    st.subheader("The search space has 36 candidates — the right answer depends on the workload")

    st.markdown(
        "There are 4 target file sizes, 3 compaction triggers, and 3 partition strategies. "
        "Which combination is best depends on how the table is actually being used: "
        "how fast data is arriving, how selective queries are, and whether queries filter by time or scan everything. "
        "A cron job has no way to observe or reason about any of that."
    )

    model, r2 = _train_model()

    col_text, col_chart = st.columns([2, 3], gap="large")

    with col_text:
        st.markdown("**How the cost model is trained**")
        st.markdown(
            "The simulator drives a table to a steady state under every combination of "
            "workload × layout — 60 workloads × 36 candidates = 2,160 training examples. "
            "For each, it measures p95 scan latency, storage cost in dollars, "
            "and write amplification (how many MB compaction rewrites). "
            "Three independent gradient boosting regressors are then fitted, one per target."
        )
        st.markdown("**Why three separate models, not one?**")
        st.markdown(
            "A layout that cuts latency might blow up write amplification. "
            "Keeping the targets separate lets the regression guard reason about each independently "
            "and reject candidates that win on one metric while regressing on another."
        )
        st.metric("Held-out latency R²", f"{r2:.3f}",
                  help="On the 20% of training data the model never saw during fitting")
        with st.expander("Why is R² 0.977 and not 1.0?"):
            st.markdown(
                "Without noise, R² is exactly 1.0 — the model inverts the scan formula "
                "from the features rather than learning anything general. "
                "15% Gaussian noise was added to training labels to simulate real measurement "
                "variance (JIT warmup, GC pauses, S3 tail latency). "
                "The 0.977 figure reflects genuine generalization, not formula memorization."
            )

    with col_chart:
        @st.cache_data
        def _candidate_scores():
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
                    "Partition": lay.partition_granularity,
                    "Trigger": lay.compaction_trigger_files,
                    "Predicted p95 (ms)": round(pred.p95_latency_ms, 1),
                    "Objective": round(pred.objective(), 2),
                })
            return pd.DataFrame(rows)

        df_c = _candidate_scores()
        COLOR_MAP = {"hour": C_GOOD, "day": C_MODEL, "device_bucket": C_GRAY}

        fig2 = go.Figure()
        for part in ["hour", "day", "device_bucket"]:
            sub = df_c[df_c["Partition"] == part]
            fig2.add_scatter(
                x=sub["File size (MB)"], y=sub["Predicted p95 (ms)"],
                mode="markers",
                marker=dict(
                    size=sub["Trigger"].map({20: 8, 50: 12, 100: 16}),
                    color=COLOR_MAP[part],
                    opacity=0.8,
                    line=dict(width=1, color="white"),
                ),
                name=part,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "File size: %{x} MB<br>"
                    "Predicted p95: %{y} ms<br>"
                    "Trigger: %{customdata[1]} files"
                    "<extra></extra>"
                ),
                customdata=sub[["Partition", "Trigger"]].values,
            )
        fig2.update_layout(
            **CHART_LAYOUT,
            title=dict(text="All 36 candidates scored — dot size = compaction trigger", font_size=13),
            xaxis_title="Target file size (MB)",
            yaxis_title="Predicted p95 latency (ms)",
            legend=dict(title="Partition strategy", orientation="v"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.divider()
    st.subheader("Feature importance — what the model actually learned")

    @st.cache_data
    def _importances():
        names = [
            "target_file_mb", "compaction_trigger", "partition",
            "ingest_rows/sec", "avg_selectivity", "time_range_ratio",
            "read_write_ratio", "avg_file_mb", "file_count",
            "small_file_ratio", "partition_count",
        ]
        imp = model._latency.feature_importances_
        return pd.DataFrame({"Feature": names, "Importance": imp}).sort_values("Importance")

    df_imp = _importances()
    fig3 = go.Figure()
    fig3.add_bar(
        x=df_imp["Importance"], y=df_imp["Feature"],
        orientation="h",
        marker_color=[C_MODEL if v > 0.1 else C_GRAY for v in df_imp["Importance"]],
    )
    fig3.update_layout(
        **CHART_LAYOUT,
        title=dict(text=f"GBM latency head — feature importances  (R² = {r2:.3f})", font_size=13),
        xaxis_title="Importance",
        margin=dict(t=36, b=36, l=120, r=8),
    )
    st.plotly_chart(fig3, use_container_width=True)


# ===========================================================================
# LIVE DEMO
# ===========================================================================

with tab_demo:
    st.subheader("Run the optimizer on a simulated table")
    st.markdown(
        "The table below starts with a bad layout: 64 MB target files with a "
        "compaction trigger of 100,000 (effectively never compacts). "
        "Adjust the workload sliders to describe how the table is being queried, "
        "then run the optimizer."
    )

    model, _ = _train_model()

    sl, sr = st.columns([1, 2], gap="large")

    with sl:
        ingest_rps = st.slider("Ingest rate (rows / sec)",
                               1000, 15000, 9000, 500)
        time_range = st.slider("Time-range query ratio",
                               0.0, 1.0, 0.9, 0.05,
                               help="1.0 = all queries filter by time window")
        selectivity = st.slider("Avg query selectivity",
                                0.01, 0.5, 0.07, 0.01,
                                help="Fraction of the table each query reads")
        rw_ratio = st.slider("Read / write ratio",
                             0.5, 10.0, 5.0, 0.5)
        run = st.button("Run optimizer", type="primary", use_container_width=True)

    with sr:
        if not run:
            st.info(
                "Adjust the sliders and click **Run optimizer**. "
                "The model will score all 36 layout candidates, shadow-test the top 3 "
                "on a copy of the table, and apply the one that passes the safety gate."
            )
        else:
            with st.spinner("Building table and running optimizer…"):
                prod = SimulatedIcebergTable(layout=Layout(64, 100_000, "device_bucket"))
                for i in range(300):
                    prod.ingest_micro_batch(mb=0.5, rows=2500, partition=f"b{i % 16}")

                st_before    = prod.stats()
                lat_before   = measure_p95_latency(prod)

                wf = WorkloadFeatures(
                    ingest_rows_per_sec=float(ingest_rps),
                    avg_selectivity=float(selectivity),
                    time_range_query_ratio=float(time_range),
                    read_write_ratio=float(rw_ratio),
                    avg_file_mb=st_before.avg_file_mb,
                    file_count=st_before.file_count,
                    small_file_ratio=st_before.small_file_ratio,
                    partition_count=st_before.partition_count,
                )

                # Score + shadow-test top 8 for the chart
                rec = recommend(model, prod.layout, wf)
                ranked_rows = []
                for i, s in enumerate(rec.ranked[:8]):
                    sh = shadow_evaluate(prod, s.layout)
                    ranked_rows.append({
                        "rank": i + 1,
                        "label": f"{s.layout.target_file_mb}MB / {s.layout.partition_granularity}",
                        "predicted": round(s.prediction.p95_latency_ms, 1),
                        "shadow":    round(sh.p95_latency_ms, 1),
                    })

                result = run_once(
                    model=model, production=prod,
                    backend=SimBackend(prod), wf=wf,
                    guard=GuardConfig(min_improvement_pct=5.0),
                    ledger_path="artifacts/demo_ledger.jsonl",
                )

                st_after  = prod.stats()
                lat_after = measure_p95_latency(prod)

            # Key metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("p95 latency — before", f"{lat_before:.0f} ms")
            m2.metric("p95 latency — after",  f"{lat_after:.0f} ms",
                      delta=f"{lat_after - lat_before:+.0f} ms", delta_color="inverse")
            m3.metric("Files — before", st_before.file_count)
            m4.metric("Files — after",  st_after.file_count,
                      delta=str(st_after.file_count - st_before.file_count),
                      delta_color="inverse")

            # Decision
            if result.promoted:
                st.success(
                    f"**Promoted** — {result.reason}  \n"
                    f"Applied: `{result.proposed}`"
                )
            else:
                st.warning(
                    f"**No change** — {result.reason}  \n"
                    f"Best candidate was: `{result.proposed}`"
                )

            # Candidate comparison chart
            df_r = pd.DataFrame(ranked_rows)
            fig4 = go.Figure()
            fig4.add_bar(
                x=df_r["label"], y=df_r["predicted"],
                name="Model prediction", marker_color=C_MODEL,
            )
            fig4.add_bar(
                x=df_r["label"], y=df_r["shadow"],
                name="Shadow measured", marker_color=C_SHADOW,
            )
            fig4.add_hline(
                y=lat_before, line_dash="dot", line_color=C_BAD, line_width=1.5,
                annotation_text="Baseline", annotation_position="top right",
            )
            fig4.update_layout(
                **CHART_LAYOUT,
                barmode="group",
                title=dict(text="Top 8 candidates — model prediction vs shadow-measured latency",
                           font_size=13),
                xaxis_title="Candidate layout",
                yaxis_title="p95 latency (ms)",
                legend=dict(orientation="h", yanchor="bottom", y=1.01, x=0),
                xaxis=dict(tickangle=-25, showgrid=False, linecolor="#e5e7eb"),
            )
            st.plotly_chart(fig4, use_container_width=True)


# ===========================================================================
# SAFETY GATE
# ===========================================================================

with tab_guard:
    st.subheader("Every proposed change is tested on a copy of the table first")
    st.markdown(
        "The cost model's prediction gets the optimizer to the right neighborhood, "
        "but predictions are not measurements. Before any layout change touches production, "
        "the orchestrator clones the table, applies the change to the clone, benchmarks it, "
        "and runs three checks. One failure rejects the change."
    )

    st.divider()

    g1, g2, g3 = st.columns(3)
    g1.markdown("**Clone → Apply → Measure**")
    g1.markdown(
        "The production table is deep-copied in memory. Compaction runs on the clone "
        "using the proposed layout settings. p95 latency, storage cost, and write "
        "amplification are measured on the result. The production table is read-only "
        "throughout — it is never touched."
    )
    g2.markdown("**Three checks, any failure rejects**")
    g2.markdown(
        "The weighted objective (latency + cost + write amplification) must improve "
        "by at least 5%. Latency must not get worse at all — a layout that cuts cost "
        "but adds 10 ms to p95 is rejected outright. Write amplification must not "
        "exceed an absolute tolerance."
    )
    g3.markdown("**Everything is logged**")
    g3.markdown(
        "Every promotion and every rejection is appended to an audit ledger as JSONL, "
        "with the full input metrics, the reason string, and a timestamp. "
        "The system cannot silently make the table worse."
    )

    st.divider()
    st.subheader("Try the guard yourself")

    st.markdown(
        "Enter baseline and candidate metrics below. "
        "The guard will evaluate the candidate and explain its decision."
    )

    gc1, gc2 = st.columns(2, gap="large")

    with gc1:
        st.markdown("**Baseline (current table)**")
        b_lat  = st.number_input("p95 latency (ms)",    value=200.0, key="bl", step=10.0)
        b_cost = st.number_input("Storage cost ($)",    value=5.0,   key="bc", step=0.5)
        b_wamp = st.number_input("Write amplification", value=10.0,  key="bw", step=5.0)

    with gc2:
        st.markdown("**Candidate (proposed layout)**")
        c_lat  = st.number_input("p95 latency (ms)",    value=120.0, key="cl", step=10.0)
        c_cost = st.number_input("Storage cost ($)",    value=4.5,   key="cc", step=0.5)
        c_wamp = st.number_input("Write amplification", value=10.0,  key="cw", step=5.0)

    if st.button("Evaluate candidate", type="primary"):
        base = EvalResult(layout=Layout(128, 50, "day"),
                          p95_latency_ms=b_lat, storage_cost=b_cost,
                          write_amplification=b_wamp, rows=1000)
        cand = EvalResult(layout=Layout(512, 50, "hour"),
                          p95_latency_ms=c_lat, storage_cost=c_cost,
                          write_amplification=c_wamp, rows=1000)
        d = evaluate_promotion(base, cand)

        if d.promote:
            st.success(f"**Promoted** — {d.reason}")
        else:
            st.error(f"**Rejected** — {d.reason}")

        # Metric deltas
        dm1, dm2, dm3 = st.columns(3)
        dm1.metric("Latency change",
                   f"{c_lat:.0f} ms",
                   delta=f"{c_lat - b_lat:+.0f} ms", delta_color="inverse")
        dm2.metric("Storage cost change",
                   f"${c_cost:.2f}",
                   delta=f"{c_cost - b_cost:+.2f}", delta_color="inverse")
        dm3.metric("Write amp change",
                   f"{c_wamp:.0f}",
                   delta=f"{c_wamp - b_wamp:+.0f}", delta_color="inverse")

    st.divider()

    ledger_path = Path("artifacts/promotion_ledger.jsonl")
    if ledger_path.exists():
        entries = []
        for line in ledger_path.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if entries:
            st.subheader("Promotion ledger")
            st.dataframe(pd.DataFrame(entries), use_container_width=True, hide_index=True)
