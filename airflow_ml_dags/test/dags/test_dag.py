from unittest.mock import patch

import pytest
from airflow.models import DagBag

UPLOAD_DAG_STRUCTURE = {"docker-airflow-upload": set()}


TRAIN_DAG_STRUCTURE = {
    "wait_for_train_data": {"docker-airflow-preprocess-train"},
    "wait_for_train_target": {"docker-airflow-preprocess-train"},
    "docker-airflow-preprocess-train": {"docker-airflow-split"},
    "docker-airflow-split": {"docker-airflow-train"},
    "docker-airflow-train": {"docker-airflow-validate"},
    "docker-airflow-validate": set(),
}

PREDICT_DAG_STRUCTURE = {
    "wait_for_prediction_data": {"docker-airflow-preprocess-valid"},
    "wait_for_prediction_model": {"docker-airflow-preprocess-valid"},
    "docker-airflow-preprocess-valid": {"docker-airflow-predict"},
    "docker-airflow-predict": set(),
}


@pytest.fixture()
@patch("airflow.models.Variable.get")
def dag_bag(var_patch):
    var_patch.return_value = ""
    return DagBag(dag_folder="./dags", include_examples=False)


def _test_dag_structure(dag_tasks, ground_truth):
    for name, task in dag_tasks.items():
        assert task.downstream_task_ids == set(ground_truth[name])


def test_dag_import(dag_bag):
    assert not dag_bag.import_errors


def test_upload_dag(dag_bag):
    upload_dag = dag_bag.dags["upload"]
    upload_dag_tasks = upload_dag.task_dict
    assert upload_dag.schedule_interval == "@daily", "Wrong schedule interval"
    assert len(upload_dag_tasks) == 1
    assert "docker-airflow-upload" in upload_dag_tasks
    _test_dag_structure(upload_dag_tasks, UPLOAD_DAG_STRUCTURE)


def test_train_dag(dag_bag):
    train_dag = dag_bag.dags["train"]
    train_dag_tasks = train_dag.task_dict
    assert train_dag.schedule_interval == "@weekly", "Wrong schedule interval"
    assert len(train_dag_tasks) == 6
    assert "wait_for_train_data" in train_dag_tasks
    assert "wait_for_train_target" in train_dag_tasks
    assert "docker-airflow-preprocess-train" in train_dag_tasks
    assert "docker-airflow-split" in train_dag_tasks
    assert "docker-airflow-train" in train_dag_tasks
    assert "docker-airflow-validate" in train_dag_tasks
    _test_dag_structure(train_dag_tasks, TRAIN_DAG_STRUCTURE)


def test_predict_dag(dag_bag):
    predict_dag = dag_bag.dags["predict"]
    predict_dag_tasks = predict_dag.task_dict
    assert predict_dag.schedule_interval == "@daily", "Wrong schedule interval"
    assert len(predict_dag_tasks) == 4
    assert "wait_for_prediction_data" in predict_dag_tasks
    assert "wait_for_prediction_model" in predict_dag_tasks
    assert "docker-airflow-preprocess-valid" in predict_dag_tasks
    assert "docker-airflow-predict" in predict_dag_tasks
    _test_dag_structure(predict_dag_tasks, PREDICT_DAG_STRUCTURE)
