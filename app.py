import base64
import csv
import datetime
import os
import numpy as np
import tensorflow as tf
import cv2
from flask import Flask, request, render_template, jsonify
from io import BytesIO
from PIL import Image

from services.top_predict import get_top_predictions, plot_prediction_chart

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model("traffic_model.h5")
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

segmentation_model = tf.keras.models.load_model("segmentation_model.h5")
segmentation_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define class names
CLASS_NAMES = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)',
    'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)',
    'Speed limit (120km/h)', 'No passing', 'No passing veh over 3.5 tons', 'Right-of-way at intersection',
    'Priority road', 'Yield', 'Stop', 'No vehicles', 'Veh > 3.5 tons prohibited', 'No entry',
    'General caution', 'Dangerous curve left', 'Dangerous curve right', 'Double curve', 'Bumpy road',
    'Slippery road', 'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians',
    'Children crossing', 'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing',
    'End speed + passing limits', 'Turn right ahead', 'Turn left ahead', 'Ahead only',
    'Go straight or right', 'Go straight or left', 'Keep right', 'Keep left', 'Roundabout mandatory',
    'End of no passing', 'End no passing veh > 3.5 tons'
]

def image_to_base64(img):
    buffered = BytesIO()
    if isinstance(img, np.ndarray):
        # Ensure array is in uint8 and has 3 channels
        img = (img * 255).astype(np.uint8)
        if len(img.shape) == 2 or img.shape[-1] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        img = Image.fromarray(img)
    # Convert to RGB if not already
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

# Preprocess image
def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

def classify_image(file_bytes):
    # Convert bytes to numpy array
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Basic info
    height, width, channels = img.shape
    mean = cv2.mean(img)[:3]
    std_dev = cv2.meanStdDev(img)
    std_dev = std_dev[1].flatten()

    # Canny edges
    edges = cv2.Canny(img, 100, 200)

    # Hough lines
    edges_hough = cv2.Canny(img, 50, 150)
    hough_img = img.copy()
    lines = cv2.HoughLinesP(edges_hough, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Harris corners
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    harris = cv2.cornerHarris(gray, 2, 3, 0.04)
    harris_img = img.copy()
    harris_img[harris > 0.01 * harris.max()] = [0, 0, 255]

    # Contours
    contours_img = img.copy()
    gray_for_contours = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray_for_contours, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(contours_img, contours, -1, (0, 255, 0), 2)

    # Convert images to base64
    edges_b64 = image_to_base64(edges)
    hough_b64 = image_to_base64(hough_img)
    harris_b64 = image_to_base64(harris_img)
    contours_b64 = image_to_base64(contours_img)

    result = {
        "image_info": {
            "shape": {
                "height": height,
                "width": width,
                "channels": channels
            },
            "mean_values": {
                "red": float(mean[2]),
                "green": float(mean[1]),
                "blue": float(mean[0]),
                "gray": float(cv2.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[0])
            },
            "std_dev_values": {
                "red": float(std_dev[2]),
                "green": float(std_dev[1]),
                "blue": float(std_dev[0]),
                "gray": float(np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)))
            },
            "edge_detection": edges_b64,
            "hough_lines": hough_b64,
            "harris_corners": harris_b64,
            "contours": contours_b64
        }
    }
    
    return result

# Home route
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    original_img_b64 = None
    processed_img_b64 = None
    chart = None
    masked_image_b64 = None
    classification_results = None

    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", error="No image uploaded")
        else:
            file = request.files["image"]
            if file.filename == "":
                return render_template("index.html", error="No image selected")
            else:
                try:
                    # Read the file once and store it
                    file_bytes = file.read()
                    
                    # Perform image classification
                    classification_results = classify_image(file_bytes)
                    
                    # Load and preprocess image for traffic sign recognition
                    image = Image.open(BytesIO(file_bytes))
                    processed_image = preprocess_image(image)  # shape: (1, H, W, C)

                    # Apply segmentation
                    mask = segmentation_model.predict(processed_image)
                    mask = (mask > 0.5).astype(np.float32)  
                    masked_image = processed_image * mask

                    # Predict traffic sign
                    pred = model.predict(masked_image)
                    predicted_class = np.argmax(pred, axis=1)[0]
                    prediction = CLASS_NAMES[predicted_class]
                    confidence = float(np.max(pred)) * 100

                    # Plot prediction bar chart
                    top_labels, top_scores = get_top_predictions(pred, CLASS_NAMES)
                    chart = plot_prediction_chart(top_labels, top_scores)

                    # Convert images to base64 for HTML
                    original_img_b64 = image_to_base64(image)
                    processed_img_b64 = image_to_base64(processed_image[0])
                    masked_image_b64 = image_to_base64(masked_image[0])
                except Exception as e:
                    return render_template("index.html", error=str(e))

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        original_image=original_img_b64,
        processed_image=processed_img_b64,
        chart=chart,
        masked_image=masked_image_b64,
        classification=classification_results
    )

@app.route("/feedback", methods=["POST"])
def feedback():
    prediction = request.form.get("prediction")
    confidence = request.form.get("confidence")
    feedback = request.form.get("feedback")
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save to CSV (or use Firestore or any DB)
    with open("feedback_log.csv", mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, prediction, confidence, feedback])

    return render_template("thank_you.html", feedback=feedback)

# API endpoint for image classification
@app.route("/api/classify", methods=["POST"])
def api_classify():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400
    
    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No image selected"}), 400
    
    try:
        file_bytes = file.read()
        result = classify_image(file_bytes)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)