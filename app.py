import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from werkzeug.utils import secure_filename
import model as ids_model  # This imports your model.py

app = Flask(__name__)
app.secret_key = "change-this"  # change in production

# Ensure upload and plot directories exist
os.makedirs("uploads", exist_ok=True)
os.makedirs(os.path.join("static", "plots"), exist_ok=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        flash("No file part.")
        return redirect(url_for("dashboard"))

    file = request.files["file"]
    if file.filename == "":
        flash("Please select a CSV file.")
        return redirect(url_for("dashboard"))

    filename = secure_filename(file.filename)
    upload_path = os.path.join("uploads", filename)
    file.save(upload_path)

    try:
        df = pd.read_csv(upload_path)
    except Exception as e:
        flash(f"Could not read CSV: {e}")
        return redirect(url_for("dashboard"))

    run_id = uuid.uuid4().hex[:8]
    results = ids_model.run_ids(df, run_id=run_id)

    return render_template(
        "results.html",
        preview_html=results["preview_html"],
        report_html=results["report_html"],
        cm_path=results["cm_path"],
        dist_path=results.get("dist_path"),
        timeline_path=results.get("timeline_path"),
        notes=results.get("notes", [])
    )

if __name__ == "__main__":
    app.run(debug=True)
