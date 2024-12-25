from Audio_Classification import logger
from Audio_Classification.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from Audio_Classification.pipeline.stage_02_base_model import PrepareBaseModelPipeline
from Audio_Classification.pipeline.stage_03_training import ModelTrainingPipeline
from Audio_Classification.pipeline.stage_04_evaluation import EvaluationPipeline 

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f"====== Stage {STAGE_NAME} Started =============")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f"====== Stage {STAGE_NAME} Completed =============")
except Exception as e:
    logger.exception(f"====== Stage {STAGE_NAME} Failed due to {e} ======")
    raise e



STAGE_NAME = "Prepare Base Model Stage"
try:
    logger.info(f"====== Stage {STAGE_NAME} Started =============")
    obj = PrepareBaseModelPipeline()
    obj.main()
    logger.info(f"====== Stage {STAGE_NAME} Completed =============")
except Exception as e:
    logger.exception(f"====== Stage {STAGE_NAME} Failed due to {e} ======")
    raise e


STAGE_NAME = "Training"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Evaluation stage"
try:
    logger.info(f"*******************")
    logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
    obj = EvaluationPipeline()
    obj.main()
    logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
    logger.exception(e)
    raise e