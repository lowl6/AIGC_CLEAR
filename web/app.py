#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Flask app for AIGC_CLEAR web showcase.

This app is organized around templates under web/templates:
- index.html
- detect.html
- result.html
- api.html
- pricing.html
- dashboard.html
"""

from flask import Flask, abort, jsonify, redirect, render_template, request, url_for
from jinja2 import TemplateNotFound

app = Flask(__name__, template_folder="templates", static_folder="static")


TEMPLATE_PAGES = {
    "index": "index.html",
    "detect": "detect.html",
    "result": "result.html",
    "api": "api.html",
    "pricing": "pricing.html",
    "dashboard": "dashboard.html",
}


def _render_page(template_name: str):
    try:
        return render_template(template_name)
    except TemplateNotFound:
        abort(404, description=f"Template not found: {template_name}")


@app.route("/")
def index():
    return _render_page(TEMPLATE_PAGES["index"])


# Keep compatibility with links like href="index.html"
@app.route("/index.html")
def index_html():
    return redirect(url_for("index"), code=302)


@app.route("/detect")
@app.route("/detect.html")
def detect_page():
    return _render_page(TEMPLATE_PAGES["detect"])


@app.route("/result")
@app.route("/result.html")
def result_page():
    return _render_page(TEMPLATE_PAGES["result"])


@app.route("/api-docs")
@app.route("/api.html")
def api_docs_page():
    return _render_page(TEMPLATE_PAGES["api"])


@app.route("/pricing")
@app.route("/pricing.html")
def pricing_page():
    return _render_page(TEMPLATE_PAGES["pricing"])


@app.route("/dashboard")
@app.route("/dashboard.html")
def dashboard_page():
    return _render_page(TEMPLATE_PAGES["dashboard"])


@app.route("/api/checkpoint_layout", methods=["GET"])
def checkpoint_layout() -> tuple:
    return jsonify(
        {
            "base_checkpoint_root": "checkpoints/",
            "expected_layout": {
                "E_IT": "checkpoints/e_it/run_YYYYMMDD_HHMMSS/",
                "E_VL": "checkpoints/e_vl/run_YYYYMMDD_HHMMSS/",
                "E_FF": "checkpoints/e_ff/run_YYYYMMDD_HHMMSS/",
                "E_SL": "checkpoints/e_sl/run_YYYYMMDD_HHMMSS/"
            },
            "note": "Each run folder should contain at least training_summary.json and model artifacts if available."
        }
    ), 200


@app.route("/api/site_map", methods=["GET"])
def site_map() -> tuple:
    return jsonify(
        {
            "pages": {
                "home": "/",
                "detect": "/detect",
                "result": "/result",
                "api_docs": "/api-docs",
                "pricing": "/pricing",
                "dashboard": "/dashboard",
            },
            "template_pages": TEMPLATE_PAGES,
        }
    ), 200


@app.route("/api/mock_detect", methods=["POST"])
def mock_detect() -> tuple:
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()

    score = min(0.95, max(0.05, (len(text) / 120.0)))
    label = "fake" if score > 0.55 else "real"

    return jsonify(
        {
            "label": label,
            "confidence": round(score, 3),
            "note": "Web showcase endpoint. Replace with your full inference pipeline if needed."
        }
    ), 200


@app.route("/health", methods=["GET"])
def health() -> tuple:
    return jsonify({"status": "ok", "service": "AIGC_CLEAR web"}), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
