from Audio_Classification.constants import *
from Audio_Classification.utils.common import read_yaml, create_directories
from Audio_Classification.entity.config_entity import (DataIngestionConfig,
                                                       BasemodelConfig,
                                                       ModelTrainingConfig,
                                                       EvaluationConfig
                                                       )


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):
        
        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])
        create_directories([config.preprocess_dir])
        create_directories([config.split_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir = config.root_dir,
            csv_path = config.csv_path,
            preprocess_dir = config.preprocess_dir,
            audio_dataset_path = config.audio_dataset_path,
            split_dir = config.split_dir,
            preprocess_data_dir = config.preprocess_data_dir ,
            train_data_dir = config.train_data_dir,
            test_data_dir = config.test_data_dir,
        )

        return data_ingestion_config
    


    def prepare_base_model_config(self) -> BasemodelConfig:
        config = self.config.prepare_base_model

        create_directories([config.root_dir])

        base_model_config = BasemodelConfig(
            root_dir = config.root_dir,
            base_model_path= config.base_model_path
        )

        return base_model_config
    

    def get_training_config(self) -> ModelTrainingConfig:
        config = self.config.training
        params = self.params

        create_directories([config.root_dir])

        trainig_data_config = ModelTrainingConfig(
            root_dir = config.root_dir,
            train_data_dir = config.train_data_dir,
            test_data_dir = config.test_data_dir,
            trained_model_path = config.trained_model_path,
            checkpoint_model_filepath = config.checkpoint_model_filepath,
            base_model_path = config.base_model_path,
            params_epochs = params.EPOCHS,
            params_batch_size = params.BATCH_SIZE,
        )

        return trainig_data_config
    

    def get_validation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/audio_classification_model.h5",
            train_data="artifacts/data_ingestion/splits/train.csv",
            test_data = "artifacts/data_ingestion/splits/test.csv",
        )
        return eval_config