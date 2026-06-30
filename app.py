"""Gradio entrypoint for the SatNOGS triage dashboard (local or HF Space).
Run: docker compose run --rm --service-ports app python app.py
"""

from satnogs_signal.service.dashboard import build_dashboard

demo = build_dashboard("triage.db")

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
