import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from Audio_Classification.entity.config_entity import BasemodelConfig
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


class BaseModel:
    def __init__(self, config: BasemodelConfig):
        self.config = config

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

        self.save_model(path=self.config.base_model_path, model=self.model)

    
    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)