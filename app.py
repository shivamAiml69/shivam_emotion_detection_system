from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
from PIL import Image


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/'


model = tf.keras.models.load_model("emotion_transfer.keras")
model.make_predict_function()  

# Emotion classes
class_labels = ['angry','disgust','fear','happy','neutral','sad','surprise']


def preprocess_image(img_path):
    img = Image.open(img_path)
    img = img.convert("RGB")  # ensure 3 channels
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.route("/")
def home():
    return render_template("index.html")  # simple upload form

@app.route("/uploader", methods=['POST'])
def uploader():
    if 'file1' not in request.files:
        return "No file uploaded"
    
    f = request.files['file1']
    filename = secure_filename(f.filename)
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    f.save(save_path)

    # Preprocess and predict
    img = preprocess_image(save_path)
    pred = model.predict(img, verbose=0)
    predicted_class = int(np.argmax(pred))
    confidence = float(np.max(pred)) * 100

    result = f"{class_labels[predicted_class]} ({confidence:.2f}%)"

    return render_template("uploaded.html", sign_name=result, input_image=save_path)

if __name__ == "__main__":
    app.run(debug=False, use_reloader=False)
