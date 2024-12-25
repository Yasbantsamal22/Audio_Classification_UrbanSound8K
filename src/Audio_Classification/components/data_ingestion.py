import os
import pandas as pd 
import urllib.request as request
import zipfile
from Audio_Classification import logger
from Audio_Classification.utils.common import *
import numpy as np
from tqdm import tqdm
import librosa
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from Audio_Classification.entity.config_entity import DataIngestionConfig


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config


    def get_csv_file(self):
        self.metadata = pd.read_csv(self.config.csv_path)
        
        logger.info("Reading CSV File Completed")


    def feature_extractor(self,file_name):
        audio, sample_rate = librosa.load(file_name)
        mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

        return mfccs_scaled_features
    

    def get_extracted_feature(self):
        self.extracted_features = []
        for index_num, row in tqdm(self.metadata.iterrows()):
            file_name = str(os.path.join(os.path.abspath(self.config.audio_dataset_path), f'fold{row["fold"]}', row["slice_file_name"]))
            final_class_labels = row["class"]
            data = self.feature_extractor(file_name)
            self.extracted_features.append([data, final_class_labels])
    
        logger.info("Extraction of MFCC Features Completed")


    def convert_mfcc_to_dataframe(self):
        self.mfcc_df = pd.DataFrame(self.extracted_features, columns=["mfccs","class"])

        logger.info("Converted The Extracted Feature and Classes to DataFrame")


    def encoding_class_variables(self):
        le = LabelEncoder()
        self.mfcc_df['class'] = le.fit_transform(self.mfcc_df['class'])

        logger.info("Encoded the Objective Data of Classes variable Using LabelEncoder")


    def save_mfcc_dataframe(self):
        self.mfcc_df.to_csv(self.config.preprocess_data_dir, index=False)

        logger.info("Saved the DataFrame as CSV File")
    

    def splitting_data_to_train_test(self):
        train_df, test_df = train_test_split(self.mfcc_df, test_size=0.2, random_state=42)
        self.train_df = train_df
        self.test_df = test_df
        self.train_df.to_csv(self.config.train_data_dir, index=False)
        self.test_df.to_csv(self.config.test_data_dir, index=False)
        logger.info("Splitted the data into train and test")

    
    