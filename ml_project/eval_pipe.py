import argparse
import logging

import numpy as np
import pandas as pd

from src import add_zero_features, deserialize_pipe, get_column_order, predict_proba
from src.configs import EvaluationParams, read_evaluation_pipeline_params
from src.constants import ARTIFACT_DIR, DATA_DIR, PROCEED_DIR

logger = logging.getLogger("ml_project")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
file_handler = logging.FileHandler("logs/eval.log")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
stdout_handler = logging.StreamHandler()
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.addHandler(stdout_handler)


def eval_pipeline(params: EvaluationParams):
    """
    Pipeline for get prediction from data
    :param params: parameters for evaluation
    :return: Nothing
    """

    logger.info("Loading data")
    data = pd.read_csv(DATA_DIR / params.raw_data)
    logger.info("Data size is %s" % data.shape[0])

    logger.info("Loading model")
    model = deserialize_pipe(ARTIFACT_DIR / params.model / "model.pkl")

    logger.info("Add zero features")
    data = add_zero_features(data, model.feature_params.zero_cols)
    column_order = get_column_order(model.feature_params)

    logger.info("Start prediction")
    predictions = predict_proba(model.pipeline, data[column_order])

    data["Prediction"] = (predictions > params.threshold).astype(np.uint8)

    logger.info("Saving predictions")
    data.to_csv(PROCEED_DIR / params.proceed_data, index=False)


def main():
    """
    Wrapper for arguments reading and start evaluation
    :return: Nothing
    """
    parser = argparse.ArgumentParser(prog="script for taking predictions from data")
    parser.add_argument(
        "--config", dest="config_path", help="path to pipeline config", required=True
    )
    args = parser.parse_args()
    params = read_evaluation_pipeline_params(args.config_path)
    eval_pipeline(params)


if __name__ == "__main__":
    main()
