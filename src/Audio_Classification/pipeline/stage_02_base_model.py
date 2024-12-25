from Audio_Classification.config.configuration import ConfigurationManager
from Audio_Classification.components.base_model import BaseModel
from Audio_Classification import logger


STAGE_NAME = "Prepare Base Model Stage"


class PrepareBaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.prepare_base_model_config()
        prepare_base_model = BaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()
        

if __name__ == '__main__':
    try:
        logger.info(f"====== Stage {STAGE_NAME} Started =============")
        obj = PrepareBaseModelPipeline()
        obj.main()
        logger.info(f"====== Stage {STAGE_NAME} Completed =============")
    except Exception as e:
        logger.exception(f"====== Stage {STAGE_NAME} Failed due to {e} ======")
        raise e