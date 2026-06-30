"""Read-only Gradio dashboard: triage queue + monitoring, over the predictions store."""

from __future__ import annotations

from satnogs_signal.service import store

SATNOGS_OBS_URL = "https://network.satnogs.org/observations/{obs_id}/"


def triage_rows(conn, limit: int = 50) -> list:
    rows = store.ranked(conn, limit)
    return [
        [
            r["obs_id"],
            SATNOGS_OBS_URL.format(obs_id=r["obs_id"]),
            r["norad"],
            r["mode"],
            round(r["p_signal"], 3),
        ]
        for r in rows
    ]


def monitoring_stats(conn) -> dict:
    return store.stats(conn)


def build_dashboard(store_path: str):
    import gradio as gr

    def _triage():
        conn = store.connect(store_path)
        try:
            return triage_rows(conn)
        finally:
            conn.close()

    def _stats_md():
        conn = store.connect(store_path)
        try:
            s = monitoring_stats(conn)
        finally:
            conn.close()
        return (
            f"**Scored observations:** {s['n']}  \n"
            f"**Predicted with-signal:** {s['n_predicted_signal']}  \n"
            f"**Mean P(signal):** {s['avg_p_signal']:.3f}"
        )

    with gr.Blocks(title="SatNOGS signal triage") as demo:
        gr.Markdown(
            "# SatNOGS signal triage (read-only)\n"
            "Unvetted observations ranked by the model's P(signal). This tool *suggests* — "
            "click an observation to vet it on SatNOGS yourself."
        )
        with gr.Tab("Triage queue"):
            gr.Dataframe(
                headers=["obs_id", "SatNOGS link", "norad", "mode", "P(signal)"],
                value=_triage,
                interactive=False,
                wrap=True,
            )
        with gr.Tab("Monitoring"):
            gr.Markdown(_stats_md)
    return demo
