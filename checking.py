from flask import Flask, request, jsonify, send_file, render_template
from ultralytics import YOLO
import cv2
import numpy as np
import tempfile
import os

app = Flask(__name__)

# Load your trained YOLOv8 model
model = YOLO("best (2).pt")


# =========================
# HOME PAGE (UI)
# =========================
@app.route("/")
def home():
    return render_template("app.html")


# =========================
# DETECTION API
# =========================
@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    results = model(img, conf=0.25)
    annotated = results[0].plot()

    boxes = results[0].boxes
    count = len(boxes)
    avg_conf = float(boxes.conf.mean()) * 100 if count > 0 else 0

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp_file.name, annotated)

    return jsonify({
        "count": count,
        "confidence": round(avg_conf, 2),
        "image_path": temp_file.name
    })


# =========================
# RETURN IMAGE
# =========================
@app.route("/image")
def get_image():
    path = request.args.get("path")
    if not path or not os.path.exists(path):
        return "Image not found", 404
    return send_file(path, mimetype="image/jpeg")


# =========================
# RUN SERVER
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

