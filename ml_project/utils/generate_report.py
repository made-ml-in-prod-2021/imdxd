import logging


def setup_logger(logger, filename):
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(name) - %(asctime)s - %(levelname)s: %(message)s')
    file_handler = logging.FileHandler(f"logs/{filename}")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    stdout_handler = logging.StreamHandler()
    stdout_handler.setLevel(logging.WARNING)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stdout_handler)
