import os
import uuid
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from predict import predict_image, class_names

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp", "webp"}

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def home():
    return render_template("index.html", class_names=class_names)

@app.route("/predict", methods=["POST"])
def run_predict():
    if "file" not in request.files:
        return render_template("index.html", class_names=class_names, error="No file uploaded")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", class_names=class_names, error="No file selected")

    if not allowed_file(file.filename):
        return render_template("index.html", class_names=class_names, error="Invalid file type")

    filename = secure_filename(file.filename)
    unique_name = str(uuid.uuid4()) + "_" + filename
    filepath = os.path.join(UPLOAD_FOLDER, unique_name)
    file.save(filepath)

    image_url = "/" + filepath.replace("\\", "/")

    try:
        label, confidence, preds = predict_image(filepath)

        return render_template(
            "result.html",
            class_names=class_names,
            prediction=label,
            confidence=round(confidence * 100, 2),
            image_path=image_url,
            error=None
        )

    except Exception as e:
        return render_template(
            "index.html",
            class_names=class_names,
            error=str(e)
        )

@app.route("/status")
def status():
    return jsonify({"status": "running"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

