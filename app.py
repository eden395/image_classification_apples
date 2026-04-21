import os
from io import BytesIO

import cv2
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from PIL import Image
from scipy.stats import kurtosis, skew

try:
    from skimage.feature import graycomatrix, graycoprops
except ImportError:  # pragma: no cover
    from skimage.feature import greycomatrix as graycomatrix, greycoprops as graycoprops

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit


class ArtifactMissingError(RuntimeError):
    pass


def load_artifact(name: str):
    path = os.path.join(ARTIFACTS_DIR, name)
    if not os.path.exists(path):
        raise ArtifactMissingError(
            f"Missing artifact: {name}. Export the trained files into the artifacts/ folder before deploying."
        )
    return joblib.load(path)


def load_pipeline_assets():
    model = load_artifact("model.joblib")
    imputer = load_artifact("imputer.joblib")
    scaler = load_artifact("scaler.joblib")
    selector = load_artifact("selector.joblib")
    label_encoder = load_artifact("label_encoder.joblib")
    feature_columns = load_artifact("feature_columns.joblib")
    selected_features = None
    selected_path = os.path.join(ARTIFACTS_DIR, "selected_features.joblib")
    if os.path.exists(selected_path):
        selected_features = joblib.load(selected_path)
    return model, imputer, scaler, selector, label_encoder, feature_columns, selected_features


def pil_to_bgr(file_storage) -> np.ndarray:
    image = Image.open(BytesIO(file_storage.read())).convert("RGB")
    rgb = np.array(image)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr


def extract_features_from_image(img_bgr: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    img_bgr = cv2.resize(img_bgr, (128, 128))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    feats: dict[str, float] = {}

    for space_name, img, channels in [
        ("rgb", img_rgb, ["r", "g", "b"]),
        ("hsv", img_hsv, ["h", "s", "v"]),
        ("lab", img_lab, ["l", "a", "b"]),
    ]:
        for idx, ch in enumerate(channels):
            ch_vals = img[:, :, idx].astype(np.float64).ravel()
            feats[f"{space_name}_{ch}_mean"] = float(ch_vals.mean())
            feats[f"{space_name}_{ch}_std"] = float(ch_vals.std())
            feats[f"{space_name}_{ch}_skewness"] = float(np.nan_to_num(skew(ch_vals), nan=0.0))
            feats[f"{space_name}_{ch}_kurtosis"] = float(np.nan_to_num(kurtosis(ch_vals), nan=0.0))

    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    for prop in ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]:
        feats[f"glcm_{prop}"] = float(graycoprops(glcm, prop)[0, 0])

    edges = cv2.Canny(gray, 100, 200)
    feats["edge_density"] = float(edges.sum()) / (128 * 128 * 255)

    row = {col: feats.get(col, 0.0) for col in feature_cols}
    return pd.DataFrame([row])


@app.route("/", methods=["GET", "POST"])
def index():
    context = {
        "prediction": None,
        "confidence": None,
        "filename": None,
        "error": None,
        "selected_features_count": None,
        "class_probabilities": None,
    }

    try:
        model, imputer, scaler, selector, label_encoder, feature_columns, selected_features = load_pipeline_assets()
        context["selected_features_count"] = len(selected_features) if selected_features is not None else None
    except ArtifactMissingError as exc:
        context["error"] = str(exc)
        return render_template("index.html", **context)

    if request.method == "POST":
        file = request.files.get("image")
        if file is None or file.filename == "":
            context["error"] = "Please upload an apple image first."
            return render_template("index.html", **context)

        try:
            img_bgr = pil_to_bgr(file)
            features_df = extract_features_from_image(img_bgr, feature_columns)
            x_imp = imputer.transform(features_df)
            x_scaled = scaler.transform(x_imp)
            x_sel = selector.transform(x_scaled)

            pred_encoded = int(model.predict(x_sel)[0])
            pred_label = label_encoder.inverse_transform([pred_encoded])[0]

            context["prediction"] = pred_label.title()
            context["filename"] = file.filename

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(x_sel)[0]
                context["confidence"] = round(float(probs[pred_encoded]) * 100, 2)
                context["class_probabilities"] = [
                    {"label": label_encoder.classes_[i].title(), "value": round(float(prob) * 100, 2)}
                    for i, prob in enumerate(probs)
                ]
        except Exception as exc:  # pragma: no cover
            context["error"] = f"Prediction failed: {exc}"

    return render_template("index.html", **context)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
