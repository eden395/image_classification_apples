from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import cv2
import joblib

app = Flask(__name__)

# =========================
# LOAD MODEL FILES
# =========================
model = joblib.load("artifacts/model.joblib")
scaler = joblib.load("artifacts/scaler.joblib")
imputer = joblib.load("artifacts/imputer.joblib")
selector = joblib.load("artifacts/selector.joblib")
le = joblib.load("artifacts/label_encoder.joblib")


# =========================
# FEATURE EXTRACTION
# =========================
def extract_features_from_file(file):
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    img = cv2.resize(img, (128, 128))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # GREEN
    lower_green = np.array([25, 40, 40])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    # RED
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])

    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # ROTTEN
    lower_brown = np.array([10, 100, 20])
    upper_brown = np.array([30, 255, 200])

    lower_dark = np.array([0, 0, 0])
    upper_dark = np.array([180, 255, 100])

    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)
    dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)

    rotten_mask = cv2.bitwise_or(brown_mask, dark_mask)

    total_pixels = 128 * 128

    green_ratio = np.sum(green_mask > 0) / total_pixels
    red_ratio = np.sum(red_mask > 0) / total_pixels
    rotten_ratio = np.sum(rotten_mask > 0) / total_pixels

    return [green_ratio, red_ratio, rotten_ratio]


# =========================
# PREDICTION FUNCTION
# =========================
def predict_image(file):
    features = extract_features_from_file(file)

    df = pd.DataFrame([features], columns=["green", "red", "rotten"])

    X_imp = imputer.transform(df)
    X_scaled = scaler.transform(X_imp)
    X_sel = selector.transform(X_scaled)

    prediction = model.predict(X_sel)
    label = le.inverse_transform(prediction)

    return label[0]


# =========================
# ROUTES
# =========================
@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", result="No file uploaded")

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html", result="No selected file")

        try:
            result = predict_image(file)
        except Exception as e:
            result = f"Error: {str(e)}"

    return render_template("index.html", result=result)


if __name__ == "__main__":
    app.run(debug=True)