from Audio_Classification.config.configuration import ConfigurationManager
from Audio_Classification.components.data_ingestion import DataIngestion
from Audio_Classification import logger


STAGE_NAME = "Data Ingestion Stage"


class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        data_ingestion.get_csv_file()
        data_ingestion.get_extracted_feature()
        data_ingestion.convert_mfcc_to_dataframe()
        data_ingestion.encoding_class_variables()
        data_ingestion.save_mfcc_dataframe()
        data_ingestion.splitting_data_to_train_test()
        

if __name__ == '__main__':
    try:
        logger.info(f"====== Stage {STAGE_NAME} Started =============")
        obj = DataIngestionTrainingPipeline()
        obj.main()
        logger.info(f"====== Stage {STAGE_NAME} Completed =============")
    except Exception as e:
        logger.exception(f"====== Stage {STAGE_NAME} Failed due to {e} ======")
        raise e