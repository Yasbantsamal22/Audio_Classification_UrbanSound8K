import numpy as np
import tensorflow as tf
import librosa
from tensorflow.keras.models import load_model
from Audio_Classification.components.data_ingestion import DataIngestion
from Audio_Classification.config.configuration import ConfigurationManager
import os



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        # load model
        MODEL_PATH = 'artifacts/training/audio_classification_model.h5'
        model = tf.keras.models.load_model(MODEL_PATH)


        audio, sample_rate = librosa.load(self.filename)
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_sacled_features = np.mean(mfccs_features.T, axis = 0)
        mfccs_sacled_features = mfccs_sacled_features.reshape(1,-1)
        predicted_label = model.predict(mfccs_sacled_features)

        # Convert probabilities to class indices
        predicted_indices = np.argmax(predicted_label)
        
        return predicted_indices
