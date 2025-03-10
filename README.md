# 🎧 Audio Classification using Deep Learning (UrbanSound8K)

## 📌 Overview

This project is an **Audio Classification System** that can automatically recognize and classify urban sounds like sirens, dog barks, jackhammers, and more from `.wav` audio files. The system leverages deep learning techniques, feature extraction, and a user-friendly web interface to make audio recognition accessible and practical for real-world applications.

---

## 🚀 Key Features

- 🎙️ **Audio File Upload**: Users can upload `.wav` files via a web interface.
- 🧠 **Deep Learning Model**: A Convolutional Neural Network (CNN) trained on **UrbanSound8K** dataset for multi-class classification.
- 📊 **Model Evaluation**: Tracks performance using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
- 📈 **Visualization**: Includes training and evaluation visualizations like confusion matrix heatmaps.
- 🌐 **Web Application**: Flask-based web app for real-time audio classification.
- 📂 **Result Storage**: Evaluation results and training history stored in JSON format for analysis.

---

## 🧰 Technologies Used

- **Programming Language**: Python
- **Libraries/Frameworks**: TensorFlow, Keras, Librosa, Scikit-learn, Flask, NumPy, Pandas, Matplotlib, Seaborn
- **Deployment**: Flask (for web app)
- **Data**: UrbanSound8K Dataset
- **Tools**: Jupyter Notebook, VS Code, Anaconda, Git

---

## ⚙️ How It Works

1. **Data Preprocessing**: 
   - Extracted MFCC features from audio files using Librosa.
   - Encoded class labels using Label Encoding and one-hot encoding for model training.
   - Normalized input features for better model convergence.

2. **Model Training**:
   - Designed a CNN architecture optimized for audio spectrograms.
   - Trained the model on UrbanSound8K dataset with multiple sound classes.
   - Achieved robust performance validated on test data.

3. **Model Evaluation**:
   - Evaluated with classification report (Accuracy, Precision, Recall, F1-Score).
   - Confusion Matrix plotted for visual performance analysis.
   - Stored evaluation metrics and loss history in JSON files for reproducibility.

4. **Web Deployment**:
   - Flask web application built for users to upload `.wav` files.
   - The app predicts and displays the class of uploaded audio in real time.

---

## 📊 Classes Covered

- Dog Bark
- Children Playing
- Car Horn
- Air Conditioner
- Street Music
- Gun Shot
- Siren
- Engine Idling
- Jackhammer
- Drilling

---

## 💻 How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone (https://github.com/Yasbantsamal22/Audio_Classification_UrbanSound8K.git)
   cd audio-classification
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Flask application**:
   ```bash
   python app.py
   ```

4. **Access the app**:
   - Open your browser and navigate to `http://127.0.0.1:5000/`
   - Upload your `.wav` file and get predictions!

---

## 📂 Project Structure

```
├── static/
│   └── (CSS, JS files if any)
├── templates/
│   └── index.html        # Frontend for audio upload
├── model/
│   └── audio_classification_model.h5  # Trained model
├── app.py                # Flask backend
├── evaluation.json       # Model evaluation metrics
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

---

## ✅ Results & Achievements

- Built an end-to-end audio classification system for urban sound recognition.
- Achieved high classification accuracy on multi-class audio data.
- Developed a fully functional Flask web app for user interaction.
- Visualized detailed performance metrics to analyze model behavior.
- Organized evaluation and loss data systematically for future enhancements.

---

## 🌟 Future Improvements

- Incorporate real-time streaming audio classification.
- Optimize the model for edge devices and mobile deployment.
- Expand to additional sound datasets for broader coverage.
- Improve model performance using transfer learning techniques.

---

## 📬 Contact

For any inquiries or collaborations, feel free to reach out:
- **Email**: yasbantsamal@gmail.com
