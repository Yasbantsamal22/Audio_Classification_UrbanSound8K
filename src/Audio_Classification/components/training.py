import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
import pandas as pd 
import numpy as np
import ast
import re
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from pathlib import Path
from Audio_Classification.entity.config_entity import ModelTrainingConfig
import warnings
warnings.filterwarnings("ignore")


class Training:
    def __init__(self, config: ModelTrainingConfig):
        self.config = config


    def get_train_test_data(self):
        self.train_df = pd.read_csv(self.config.train_data_dir)
        self.test_df = pd.read_csv(self.config.test_data_dir)

        X_train = self.train_df["mfccs"].apply(lambda x: np.fromstring(x.replace('\n', ' ').strip('[]'), sep=' '))
        self.X_train = np.vstack(X_train)
        self.y_train = np.array(self.train_df["class"].tolist())

        X_test = self.test_df["mfccs"].apply(lambda x: np.fromstring(x.replace('\n', ' ').strip('[]'), sep=' '))
        self.X_test = np.vstack(X_test)
        self.y_test = np.array(self.test_df["class"].tolist())

        
    def get_base_model(self):
        self.model = Sequential()

        ## First Layer
        self.model.add(Dense(100,input_shape=(40,)))
        self.model.add(Activation("relu"))
        self.model.add(Dropout(0.5))

        ## Second Layer
        self.model.add(Dense(200))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        ## Third Layer
        self.model.add(Dense(100))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))

        ## Final Layer
        self.model.add(Dense(10))
        self.model.add(Activation("softmax"))

        ## Compile the self.model
        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.model.summary()

    def train(self):
        checkpointer = ModelCheckpoint(filepath=self.config.checkpoint_model_filepath, verbose=1, save_best_only=True)

        start = datetime.now()

        self.model.fit(self.X_train, self.y_train,
                       batch_size=self.config.params_batch_size, 
                       epochs=self.config.params_epochs, 
                       validation_data=(self.X_test, self.y_test), 
                       callbacks=[checkpointer]
                    )

        duration = datetime.now() - start
        print("Training completed in time: ",duration)

    
        self.save_model(
            path=self.config.trained_model_path,
            model=self.model
        )

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
