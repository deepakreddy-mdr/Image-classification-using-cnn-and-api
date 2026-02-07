from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from google import genai
import os

app = Flask(__name__)

# Ensure uploads directory exists to prevent errors
os.makedirs("uploads", exist_ok=True)

# Load CNN model
cnn_model = tf.keras.models.load_model("model.h5")

class_names = ['Airplane','Automobile','Bird','Cat','Deer',
               'Dog','Frog','Horse','Ship','Truck']

# Gemini setup
client = genai.Client(api_key="AIzaSyDYgpKE2pUx5GeeumVkoYsiztG5AQzsZa0")

@app.route("/", methods=["GET", "POST"])
def home():
    cnn_result = ""
    gemini_result = ""
    confidence = 0  # Default confidence variable

    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["image"]
        
        if file.filename != '':
            filepath = os.path.join("uploads", file.filename)
            file.save(filepath)

            # --- CNN prediction ---
            img = cv2.imread(filepath)
            img = cv2.resize(img, (32,32))
            img = img / 255.0
            img = np.reshape(img, (1,32,32,3))

            prediction = cnn_model.predict(img)
            
            # Get the index of the highest probability
            max_index = np.argmax(prediction)
            cnn_result = class_names[max_index]
            
            # EXTRACT CONFIDENCE: Get max value * 100 and round to 2 decimals
            confidence = round(float(np.max(prediction)) * 100, 2)

            # --- Gemini prediction ---
            try:
                image = Image.open(filepath)
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=["What object is in this image? Give only one word.", image]
                )
                gemini_result = response.text.strip() # .strip() removes accidental newlines
            except Exception as e:
                gemini_result = "Error"
                print(f"Gemini Error: {e}")

    # Pass the 'confidence' variable to the template
    return render_template("index.html",
                           cnn_result=cnn_result,
                           gemini_result=gemini_result,
                           confidence=confidence)

if __name__ == "__main__":
    app.run(debug=True)