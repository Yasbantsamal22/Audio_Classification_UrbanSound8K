from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import numpy as np
from Audio_Classification.pipeline.predict import PredictionPipeline

# Initialize the Flask app
app = Flask(__name__)

# Directory for saving uploaded files
UPLOAD_FOLDER = 'artifacts/uploads'
ALLOWED_EXTENSIONS = {'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# Define class names
class_names = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
               'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren',
               'street_music']

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        
        # Preprocess and predict
        prprocess_obj = PredictionPipeline(file_path)
        predicted_indices = prprocess_obj.predict()
        predicted_class = class_names[predicted_indices]
        
        return jsonify({'prediction': predicted_class})
    else:
        return jsonify({'error': 'File must be in .wav format'}), 400

if __name__ == '__main__':
    app.run(debug=True)
