import logging

logger = logging.getLogger("panns")
logger.setLevel(logging.DEBUG)
stream = logging.StreamHandler()
stream.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s \t %(levelname)s \t %(name)s \t %(message)s')
stream.setFormatter(formatter)
logger.addHandler(stream)

logger.debug("Base 'panns' logger configured")