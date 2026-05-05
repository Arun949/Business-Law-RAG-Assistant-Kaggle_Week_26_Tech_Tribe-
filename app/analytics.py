"""
analytics.py  —  Analytics dashboard tab for the Gradio UI.

Exposes build_analytics_tab(session_log_state) which wires up all
charts and tables.  Import and call from gradio_app.py inside a
gr.Tab("📊 Analytics") block.
"""

import json
import os
import time
from datetime import datetime

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
_COST_FILE     = "data/cost_tracker.json"
_EVAL_FILE     = "data/eval_results.json"
_QUESTIONS     = "data/questions.json"
_FEEDBACK_FILE = "data/feedback_log.jsonl"
_BUDGET        = 5.00

# ── Colour palette (matches the dark Gradio theme) ────────────────────────────
_BG    = "#0D1B2A"
_CARD  = "#16283E"
_GOLD  = "#D4A017"
_CYAN  = "#4AB8C4"
_GREEN = "#2ECC71"
_RED   = "#E74C3C"
_GREY  = "#8899AA"
_WHITE = "#FFFFFF"


# ── Data loaders ───────────────────────────────────────────────────────────────

def _load_cost() -> dict:
    if os.path.exists(_COST_FILE):
        with open(_COST_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {"total_usd": 0.0, "calls": 0, "by_script": {}, "last_updated": None}


def _load_eval() -> dict:
    if os.path.exists(_EVAL_FILE):
        with open(_EVAL_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _load_questions() -> list:
    if os.path.exists(_QUESTIONS):
        with open(_QUESTIONS, encoding="utf-8") as f:
            return json.load(f)
    return []


def _load_feedback() -> list:
    entries = []
    if os.path.exists(_FEEDBACK_FILE):
        with open(_FEEDBACK_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return entries


# ── Figure builders ────────────────────────────────────────────────────────────

def _fig(w=7, h=3.5):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_CARD)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GREY)
    ax.tick_params(colors=_WHITE, labelsize=9)
    ax.xaxis.label.set_color(_WHITE)
    ax.yaxis.label.set_color(_WHITE)
    ax.title.set_color(_GOLD)
    return fig, ax


def _cost_chart() -> plt.Figure:
    data = _load_cost()
    by_script = data.get("by_script", {})
    total     = data.get("total_usd", 0.0)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    fig.patch.set_facecolor(_BG)

    # ── Left: per-script bar chart
    ax = axes[0]
    ax.set_facecolor(_CARD)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GREY)

    if by_script:
        labels = list(by_script.keys())
        values = [by_script[k] for k in labels]
        colors = [_CYAN, _GOLD, _GREEN, _RED, "#9B59B6", "#E67E22"][: len(labels)]
        bars = ax.bar(labels, values, color=colors, width=0.5, zorder=3)
        ax.set_ylim(0, max(values) * 1.35 if values else 0.01)
        ax.yaxis.grid(True, color=_GREY, linestyle="--", alpha=0.35, zorder=0)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.03,
                    f"${val:.4f}", ha="center", va="bottom",
                    color=_WHITE, fontsize=8.5, fontweight="bold")
    else:
        ax.text(0.5, 0.5, "No data yet", transform=ax.transAxes,
                ha="center", va="center", color=_GREY, fontsize=11)

    ax.set_title("API Cost by Script (USD)", color=_GOLD, fontsize=11, pad=8)
    ax.tick_params(colors=_WHITE, labelsize=9)
    for sp in ax.spines.values():
        sp.set_edgecolor(_GREY)

    # ── Right: budget gauge (horizontal bar)
    ax2 = axes[1]
    ax2.set_facecolor(_CARD)
    for sp in ax2.spines.values():
        sp.set_edgecolor(_GREY)

    pct       = min(total / _BUDGET, 1.0)
    remaining = max(_BUDGET - total, 0.0)
    bar_color = _GREEN if pct < 0.6 else (_GOLD if pct < 0.85 else _RED)

    ax2.barh(["Budget"], [_BUDGET], color=_GREY, alpha=0.25, height=0.45)
    ax2.barh(["Budget"], [total],   color=bar_color, height=0.45)
    ax2.set_xlim(0, _BUDGET)
    ax2.set_title("Budget Usage  ($5.00 limit)", color=_GOLD, fontsize=11, pad=8)
    ax2.tick_params(colors=_WHITE, labelsize=9)
    ax2.xaxis.grid(True, color=_GREY, linestyle="--", alpha=0.35)

    ax2.text(total + 0.05, 0,
             f"${total:.4f} used  ({pct*100:.1f}%)",
             va="center", color=_WHITE, fontsize=9, fontweight="bold")
    ax2.text(_BUDGET - 0.05, 0,
             f"${remaining:.4f} left",
             va="center", ha="right", color=_GREY, fontsize=8.5)

    fig.tight_layout(pad=1.8)
    return fig


def _eval_chart() -> plt.Figure:
    data = _load_eval()
    by_cat = data.get("by_category", {})
    if not by_cat:
        fig, ax = _fig(7, 3.5)
        ax.text(0.5, 0.5, "Run  python app/evaluate.py  to populate",
                transform=ax.transAxes, ha="center", va="center",
                color=_GREY, fontsize=11)
        ax.set_title("Evaluation Hit Rate by Category", color=_GOLD, fontsize=11, pad=8)
        return fig

    cats   = list(by_cat.keys())
    totals = [by_cat[c]["total"] for c in cats]
    hits   = [by_cat[c]["hits"]  for c in cats]
    rates  = [h / t if t else 0.0 for h, t in zip(hits, totals)]

    fig, ax = _fig(7, 3.8)
    x      = range(len(cats))
    colors = [_GREEN if r >= 1.0 else (_GOLD if r >= 0.7 else _RED) for r in rates]
    bars   = ax.bar(x, [r * 100 for r in rates], color=colors, width=0.5, zorder=3)
    ax.set_xticks(list(x))
    ax.set_xticklabels([c.replace("-", "\n") for c in cats], color=_WHITE, fontsize=9)
    ax.set_ylim(0, 115)
    ax.yaxis.grid(True, color=_GREY, linestyle="--", alpha=0.35, zorder=0)
    ax.set_ylabel("Hit Rate (%)", color=_WHITE, fontsize=9)
    ax.set_title("Evaluation Hit Rate by Category", color=_GOLD, fontsize=11, pad=8)

    for bar, rate, h, t in zip(bars, rates, hits, totals):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{h}/{t}\n({rate*100:.0f}%)",
                ha="center", va="bottom", color=_WHITE, fontsize=8.5, fontweight="bold")

    overall = data.get("hit_rate", 0.0)
    ax.axhline(overall * 100, color=_CYAN, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(len(cats) - 0.4, overall * 100 + 3,
            f"Overall {overall*100:.0f}%", color=_CYAN, fontsize=8)

    fig.tight_layout(pad=1.5)
    return fig


def _faithfulness_chart(session_log: list) -> plt.Figure:
    """Pie chart: faithful / unfaithful / n/a across session queries."""
    faithful   = sum(1 for q in session_log if q.get("faithful") == "faithful")
    unfaithful = sum(1 for q in session_log if q.get("faithful") == "unfaithful")
    na         = sum(1 for q in session_log if q.get("faithful") == "n/a")
    total      = faithful + unfaithful + na

    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    fig.patch.set_facecolor(_BG)
    ax.set_facecolor(_BG)

    if total == 0:
        ax.text(0.5, 0.5, "No queries yet",
                transform=ax.transAxes, ha="center", va="center",
                color=_GREY, fontsize=11)
        ax.set_title("Answer Faithfulness (Session)", color=_GOLD, fontsize=11, pad=8)
        ax.axis("off")
        return fig

    sizes  = [faithful, unfaithful, na]
    labels = [f"Faithful\n{faithful}", f"Unfaithful\n{unfaithful}", f"N/A\n{na}"]
    colors = [_GREEN, _RED, _GREY]
    non_zero = [(s, l, c) for s, l, c in zip(sizes, labels, colors) if s > 0]
    if non_zero:
        s, l, c = zip(*non_zero)
        wedges, texts, autotexts = ax.pie(
            s, labels=l, colors=c,
            autopct="%1.0f%%", startangle=90,
            textprops={"color": _WHITE, "fontsize": 9},
            wedgeprops={"linewidth": 1.5, "edgecolor": _BG},
        )
        for at in autotexts:
            at.set_color(_BG)
            at.set_fontweight("bold")

    ax.set_title("Answer Faithfulness (Session)", color=_GOLD, fontsize=11, pad=8)
    fig.tight_layout()
    return fig


def _rrf_score_chart(session_log: list) -> plt.Figure:
    """Line chart of best RRF score per query in this session."""
    fig, ax = _fig(7, 3.5)

    if not session_log:
        ax.text(0.5, 0.5, "No queries yet",
                transform=ax.transAxes, ha="center", va="center",
                color=_GREY, fontsize=11)
        ax.set_title("RRF Score per Query (Session)", color=_GOLD, fontsize=11, pad=8)
        return fig

    scores = [q.get("best_rrf", 0.0) for q in session_log]
    xs     = list(range(1, len(scores) + 1))

    ax.plot(xs, scores, color=_CYAN, linewidth=2, marker="o",
            markersize=5, markerfacecolor=_GOLD, zorder=3)
    ax.axhline(0.025, color=_RED, linestyle="--", linewidth=1,
               alpha=0.7, label="Low-conf threshold (0.025)")
    ax.fill_between(xs, scores, alpha=0.12, color=_CYAN)
    ax.set_xlim(0.5, max(len(scores) + 0.5, 2))
    ax.set_ylim(0, max(max(scores) * 1.25, 0.05))
    ax.set_xlabel("Query #", color=_WHITE, fontsize=9)
    ax.set_ylabel("Best RRF Score", color=_WHITE, fontsize=9)
    ax.set_title("RRF Score per Query (Session)", color=_GOLD, fontsize=11, pad=8)
    ax.yaxis.grid(True, color=_GREY, linestyle="--", alpha=0.35, zorder=0)
    ax.legend(facecolor=_CARD, edgecolor=_GREY, labelcolor=_WHITE, fontsize=8)
    fig.tight_layout(pad=1.5)
    return fig


# ── KPI helper ─────────────────────────────────────────────────────────────────

def _kpi_md(session_log: list) -> str:
    cost   = _load_cost()
    eval_d = _load_eval()
    fb     = _load_feedback()

    total_cost  = cost.get("total_usd", 0.0)
    total_calls = cost.get("calls", 0)
    last_upd    = cost.get("last_updated", "—")
    budget_pct  = total_cost / _BUDGET * 100

    n_queries   = len(session_log)
    avg_rrf     = (sum(q.get("best_rrf", 0) for q in session_log) / n_queries
                   if n_queries else 0.0)
    eval_score  = eval_d.get("hit_rate", 0.0) * 100 if eval_d else 0.0

    liked    = sum(1 for e in fb if e.get("liked"))
    disliked = sum(1 for e in fb if not e.get("liked"))

    bar_filled = int(budget_pct / 5)
    bar_empty  = 20 - bar_filled
    budget_bar = "🟩" * bar_filled + "⬜" * bar_empty

    cost_color  = "🟢" if budget_pct < 60 else ("🟡" if budget_pct < 85 else "🔴")
    eval_color  = "🟢" if eval_score >= 90 else ("🟡" if eval_score >= 70 else "🔴")
    return (
        f"| 💰 Total API Cost | 📊 Budget Used | {eval_color} Eval Score | 🔍 Session Queries |\n"
        f"|:---:|:---:|:---:|:---:|\n"
        f"| **${total_cost:.4f}** | **{cost_color} {budget_pct:.1f}%** | **{eval_score:.0f}%** | **{n_queries}** |\n"
        f"| {total_calls} calls · {last_upd[:10] if last_upd else '—'} "
        f"| {budget_bar} | 19 questions | avg RRF {avg_rrf:.4f} · 👍{liked} 👎{disliked} |"
    )


def _session_table_df(session_log: list) -> pd.DataFrame:
    if not session_log:
        return pd.DataFrame(columns=["#", "Question", "Category", "Best RRF", "Faithful", "Cost ($)", "Time"])
    rows = []
    for i, q in enumerate(session_log, 1):
        rows.append({
            "#":          i,
            "Question":   (q.get("question", "")[:65] + "…"
                           if len(q.get("question", "")) > 65 else q.get("question", "")),
            "Category":   q.get("category", "—"),
            "Best RRF":   f"{q.get('best_rrf', 0.0):.4f}",
            "Faithful":   q.get("faithful", "—"),
            "Cost ($)":   f"{q.get('cost', 0.0):.5f}",
            "Time":       q.get("ts", "—"),
        })
    return pd.DataFrame(rows)


def _reports_table_df() -> pd.DataFrame:
    reports = sorted(
        [f for f in os.listdir("data") if f.startswith("report_") and f.endswith(".md")],
        reverse=True,
    )
    if not reports:
        return pd.DataFrame(columns=["Filename", "Created", "Size (KB)"])
    rows = []
    for r in reports[:15]:
        path = os.path.join("data", r)
        st   = os.stat(path)
        ts   = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
        rows.append({
            "Filename":    r,
            "Created":     ts,
            "Size (KB)":   f"{st.st_size / 1024:.1f}",
        })
    return pd.DataFrame(rows)


# ── Public: build the tab ──────────────────────────────────────────────────────

def build_analytics_tab(session_log_state: gr.State):
    """
    Call inside  with gr.Tab("📊 Analytics"):
        build_analytics_tab(session_log_state)
    """

    # ── Refresh function ──────────────────────────────────────────────────────

    def refresh(session_log):
        try:
            kpi        = _kpi_md(session_log)
            cost_fig   = _cost_chart()
            eval_fig   = _eval_chart()
            faith_fig  = _faithfulness_chart(session_log)
            rrf_fig    = _rrf_score_chart(session_log)
            session_df = _session_table_df(session_log)
            reports_df = _reports_table_df()
            feedback   = _load_feedback()
            n_liked    = sum(1 for e in feedback if e.get("liked"))
            n_disliked = sum(1 for e in feedback if not e.get("liked"))
            fb_md      = (
                f"**👍 Helpful:** {n_liked}  &nbsp;|&nbsp;  "
                f"**👎 Not helpful:** {n_disliked}  &nbsp;|&nbsp;  "
                f"**Total ratings:** {len(feedback)}"
                if feedback else "*No feedback logged yet — use 👍 👎 buttons in Chat.*"
            )
            return kpi, cost_fig, eval_fig, faith_fig, rrf_fig, session_df, reports_df, fb_md
        except Exception as e:
            import traceback
            err = traceback.format_exc()
            empty_df = pd.DataFrame()
            fig_err, ax = plt.subplots(); ax.text(0.5, 0.5, f"Error: {e}", transform=ax.transAxes,
                ha="center", color="red"); fig_err.patch.set_facecolor(_BG)
            return (f"**Error loading analytics:** `{e}`",
                    fig_err, fig_err, fig_err, fig_err,
                    empty_df, empty_df,
                    f"```\n{err}\n```")

    # ── Layout ────────────────────────────────────────────────────────────────

    with gr.Column():
        with gr.Row():
            gr.Markdown("## 📊 Analytics Dashboard")
            refresh_btn = gr.Button("🔄 Refresh", variant="secondary", size="sm",
                                    scale=0, min_width=110)

        kpi_display = gr.Markdown("*Loading…*")

        gr.Markdown("---")
        gr.Markdown("### 💰 Cost Breakdown  &nbsp;·&nbsp;  📈 Eval Results")
        with gr.Row():
            cost_plot = gr.Plot(label="API Cost by Script & Budget", scale=3)
            eval_plot = gr.Plot(label="Evaluation Hit Rate by Category", scale=2)

        gr.Markdown("---")
        gr.Markdown("### 🔍 Session: Faithfulness  &nbsp;·&nbsp;  RRF Scores")
        with gr.Row():
            faith_plot = gr.Plot(label="Faithfulness Verdicts (Session)", scale=1)
            rrf_plot   = gr.Plot(label="RRF Score per Query (Session)", scale=2)

        gr.Markdown("---")
        gr.Markdown("### 📋 Session Query Log")
        session_table = gr.DataFrame(
            label="Queries this session",
            interactive=False,
            wrap=True,
        )

        gr.Markdown("---")
        gr.Markdown("### 👍 User Feedback Summary")
        feedback_md = gr.Markdown("*Loading…*")

        gr.Markdown("---")
        gr.Markdown("### 📄 Generated Reports")
        reports_table = gr.DataFrame(
            label="Approved reports in data/",
            interactive=False,
        )

    # ── Wire refresh button ───────────────────────────────────────────────────

    outputs = [kpi_display, cost_plot, eval_plot, faith_plot,
               rrf_plot, session_table, reports_table, feedback_md]

    refresh_btn.click(refresh, inputs=[session_log_state], outputs=outputs)

    # Auto-load on tab select via a dummy trigger is handled by the caller.
    # Return (refresh_fn, outputs) so the caller can also trigger on load.
    return refresh_btn, refresh, outputs
