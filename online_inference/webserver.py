import logging
from typing import Dict, List, Union

import pandas as pd
from flask import Flask, Response, request, make_response, jsonify

from src import SerializedModel, deserialize_pipe, get_column_order

NUMERIC = Union[int, float]

app = Flask(__name__)
app.config["model"] = deserialize_pipe("model.pkl")


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
stdout_handler = logging.StreamHandler()
stdout_handler.setLevel(logging.DEBUG)
stdout_handler.setFormatter(formatter)
logger.addHandler(stdout_handler)


def validate_data(data: List[Dict[str, Union[int, float]]], model) -> bool:
    real_fields = model.feature_params.real_cols
    cat_fields = model.feature_params.cat_cols
    all_fields = set(real_fields).union(set(cat_fields))
    if not isinstance(data, list):
        logger.error("Data is not a list of dict")
        return False
    if len(data) == 0:
        logger.error("Blank data")
        return False
    for i, row in enumerate(data):
        if not isinstance(row, dict):
            logger.error("Row %s is not a dict" % i)
            return False
        fields_diff = all_fields - set(row.keys())
        if len(fields_diff) > 0:
            logger.error("Row %s does not contains fields: %s" % (i, " ".join(fields_diff)))
            return False
        for col in real_fields:
            if not isinstance(row[col], (int, float)):
                logger.error("%s rows key %s has wrong type %s" % (i, col, type(row[col])))
                return False
        for col in cat_fields:
            if not isinstance(row[col], (int, str)):
                logger.error("%s rows key %s has wrong type %s" % (i, col, type(row[col])))
                return False
    return True


@app.route("/predict", methods=["POST"])
def predict() -> Response:
    logger.info("Got query for prediction")
    data = request.get_json()
    model: SerializedModel = app.config["model"]
    column_order = get_column_order(model.feature_params)
    logger.info("Start data validation")
    if not validate_data(data, model):
        return make_response("Wrong data types", 400)
    logger.info("Data validation successfully")
    logger.info("Start prediction")
    predict_df = pd.DataFrame(data)
    predicts = model.pipeline.predict(predict_df[column_order])
    logger.info("Predictions were calculated")
    logger.info("Send data back to client")
    return jsonify([int(p) for p in predicts])
