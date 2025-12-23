from flask import Flask, request, jsonify, render_template
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load YOLOv8 model ONCE
model = YOLO("best (2).pt")

@app.route("/")
def home():
    return render_template("app.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]

    # Read image
    img_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

    # Run detection
    results = model(img, conf=0.25)
    annotated = results[0].plot()

    boxes = results[0].boxes
    count = len(boxes)

    avg_conf = 0
    if count > 0:
        avg_conf = float(boxes.conf.mean()) * 100

    # Encode image to base64
    _, buffer = cv2.imencode(".jpg", annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "count": count,
        "confidence": round(avg_conf, 2),
        "image": img_base64
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
