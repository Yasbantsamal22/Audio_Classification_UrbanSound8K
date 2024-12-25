from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from keras.models import load_model
from pathlib import Path
from Audio_Classification.entity.config_entity import EvaluationConfig
from Audio_Classification.utils.common import save_json

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def evaluation(self):
        # Evaluation logic here
        self.model = load_model(self.config.path_of_model)

        self.test_df = pd.read_csv(self.config.test_data)
        X_test = self.test_df["mfccs"].apply(lambda x: np.fromstring(x.replace('\n', ' ').strip('[]'), sep=' '))
        self.X_test = np.vstack(X_test)
        self.y_test = np.array(self.test_df["class"].tolist())

        self.class_names = ['dog_bark', 'children_playing', 'car_horn', 'air_conditioner',
                        'street_music', 'gun_shot', 'siren', 'engine_idling', 'jackhammer',
                        'drilling'
                    ]

        self.predictions = self.model.predict(self.X_test)
        self.predicted_classes = np.argmax(self.predictions, axis=1)
        self.true_classes = self.y_test

        # Step 2: Evaluate accuracy
        self.accuracy = accuracy_score(self.true_classes, self.predicted_classes)
        print(f"Accuracy: {self.accuracy:.2%}\n")

        # Step 3: Classification report
        self.report = classification_report(self.true_classes, self.predicted_classes, target_names=self.class_names, output_dict=True)
        print("Classification Report:\n")
        print(self.report)

        # Step 4: Confusion Matrix
        conf_matrix = confusion_matrix(self.true_classes, self.predicted_classes)

        # Step 5: Plot Confusion Matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.show()

        print(f"loss : {self.model.evaluate(self.X_test, self.y_test, verbose=0)[0]} ")

        # Save as json file
        evaluation_data = {
            "loss": self.model.evaluate(self.X_test, self.y_test, verbose=0)[0],
            "accuracy": self.accuracy,
            "classification_report": self.report,
            "confusion_matrix": conf_matrix.tolist(),
        }

        save_json(path=Path("scores.json"), data=evaluation_data)