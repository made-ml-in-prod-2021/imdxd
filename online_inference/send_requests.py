import argparse
import logging
import random
from typing import Dict, List, Union

import requests
from dataclasses import dataclass


@dataclass()
class RealValueCol:

    name: str
    mean: float
    std: float


@dataclass()
class CategoricalCol:

    name: str
    nunique: int


CATEGORICAL_COLUMNS = {
    "sex": CategoricalCol("sex", 2),
    "cp": CategoricalCol("cp", 4),
    "fbs": CategoricalCol("fbs", 2),
    "restecg": CategoricalCol("restecg", 3),
    "exang": CategoricalCol("exang", 2),
    "slope": CategoricalCol("slope", 3),
    "ca": CategoricalCol("ca", 5),
    "thal": CategoricalCol("thal", 4),
}


REAL_COLUMNS = {
    "age": RealValueCol("age", 54.366, 9.082),
    "trestbps": RealValueCol("trestbps", 131.624, 17.538),
    "chol": RealValueCol("chol", 246.264, 51.831),
    "thalach": RealValueCol("thalach", 149.647, 22.905),
    "oldpeak": RealValueCol("oldpeak", 1.04, 1.61),
}


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
stdout_handler = logging.StreamHandler()
stdout_handler.setLevel(logging.INFO)
logger.addHandler(stdout_handler)


def generate_data(size: int) -> List[Dict[str, Union[float, int]]]:
    """
    Data generation function
    :param size: size of data
    :return: dataframe with synthetic data
    """
    data = []
    for _ in range(size):
        data_dict = {}
        for name, distr in CATEGORICAL_COLUMNS.items():
            data_dict[name] = random.randint(0, distr.nunique)
        for name, distr in REAL_COLUMNS.items():
            data_dict[name] = random.normalvariate(distr.mean, distr.std)
        data.append(data_dict)
    return data


def main():
    parser = argparse.ArgumentParser(prog="script for sending prediction request")
    parser.add_argument("--number", help="number of requests", type=int, required=True)
    parser.add_argument("--url", help="url of service", type=str, required=True)
    args = parser.parse_args()
    url = args.url
    if args.url.endswith("/"):
        url = f"{url}predict"
    else:
        url = f"{url}/predict"
    for i in range(1, args.number + 1):
        logger.info("send request %s" % i)
        data = generate_data(random.randint(5, 100))
        print(data[:3])
        broken = False
        if random.randint(0, 1) == 1:
            data[0] = 1
            broken = True
        response = requests.post(url, json=data)
        logger.info("get response status %s with broken: %s" % (response.status_code, broken))


if __name__ == '__main__':
    main()
