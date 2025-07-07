from flask import Flask, render_template, request, send_from_directory
import numpy as np
import librosa
import tensorflow as tf
import joblib
import os

app = Flask(__name__)

# Load trained model and label encoder
MODEL_PATH = "model/sentiment_cnn_model.h5"
ENCODER_PATH = "model/label_encoder.pkl"

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    print("✅ Model and Label Encoder loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model or label encoder: {e}")
    exit(1)

# Ensure upload folder exists
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Extract audio features
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        return np.mean(mfccs, axis=1).reshape(1, 40, 1, 1)  # Reshaped to match model input
    except Exception as e:
        print(f"⚠️ Error extracting features: {e}")
        return None

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = None
    error_message = None
    file_url = None

    if request.method == "POST":
        if "file" not in request.files:
            error_message = "No file uploaded. Please select an audio file."
        else:
            file = request.files["file"]
            if file.filename == "":
                error_message = "No file selected. Please choose an audio file."
            else:
                file_path = os.path.join(UPLOAD_FOLDER, file.filename)
                file.save(file_path)
                file_url = f"/uploads/{file.filename}"  # Generate file URL

                # Extract features
                features = extract_features(file_path)
                if features is None:
                    error_message = "Could not process audio. Please try another file."
                else:
                    # Predict sentiment
                    prediction = model.predict(features)
                    predicted_class = np.argmax(prediction, axis=1)
                    sentiment = label_encoder.inverse_transform(predicted_class)[0]

    return render_template("index.html", sentiment=sentiment, error_message=error_message, file_url=file_url)

# Serve uploaded files
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)

if __name__ == "__main__":
    app.run(debug=True)
