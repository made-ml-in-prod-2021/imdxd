import pytest
from airflow.models import DagBag


@pytest.fixture()
def dag_bag():
    return DagBag(dag_folder="./dags", include_examples=False)


def test_dag_import(dag_bag):
    assert not dag_bag.import_errors


def test_upload_dag(dag_bag):
    upload_dag = dag_bag.dags["upload"]
    upload_dag_tasks = upload_dag.task_dict
    assert upload_dag.schedule_interval == "@daily", "Wrong schedule interval"
    assert len(upload_dag_tasks) == 1
    assert "docker-airflow-upload" in upload_dag_tasks


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
    assert len(train_dag_tasks["wait_for_train_data"].upstream_list) == 0
    assert len(train_dag_tasks["wait_for_train_target"].upstream_list) == 0
    assert len(train_dag_tasks["docker-airflow-preprocess-train"].upstream_list) == 2
    assert len(train_dag_tasks["docker-airflow-split"].upstream_list) == 1
    assert len(train_dag_tasks["docker-airflow-train"].upstream_list) == 1
    assert len(train_dag_tasks["docker-airflow-validate"].upstream_list) == 1
    assert (
            "wait_for_train_data"
            in train_dag_tasks["docker-airflow-preprocess-train"].upstream_task_ids
        )
    assert (
            "wait_for_train_target"
            in train_dag_tasks["docker-airflow-preprocess-train"].upstream_task_ids
    )
    assert (
            "docker-airflow-preprocess-train"
            in train_dag_tasks["docker-airflow-split"].upstream_task_ids
    )
    assert (
            "docker-airflow-split"
            in train_dag_tasks["docker-airflow-train"].upstream_task_ids
    )
    assert (
            "docker-airflow-train"
            in train_dag_tasks["docker-airflow-validate"].upstream_task_ids
    )


def test_predict_dag(dag_bag):
    predict_dag = dag_bag.dags["predict"]
    predict_dag_tasks = predict_dag.task_dict
    assert predict_dag.schedule_interval == "@daily", "Wrong schedule interval"
    assert len(predict_dag_tasks) == 4
    assert "wait_for_prediction_data" in predict_dag_tasks
    assert "wait_for_prediction_model" in predict_dag_tasks
    assert "docker-airflow-preprocess-valid" in predict_dag_tasks
    assert "docker-airflow-predict" in predict_dag_tasks
    assert len(predict_dag_tasks["wait_for_prediction_data"].upstream_list) == 0
    assert len(predict_dag_tasks["wait_for_prediction_model"].upstream_list) == 0
    assert len(predict_dag_tasks["docker-airflow-predict"].upstream_list) == 1
    assert len(predict_dag_tasks["docker-airflow-preprocess-valid"].upstream_list) == 2
    assert (
        "wait_for_prediction_data"
        in predict_dag_tasks["docker-airflow-preprocess-valid"].upstream_task_ids
    )
    assert (
        "wait_for_prediction_model"
        in predict_dag_tasks["docker-airflow-preprocess-valid"].upstream_task_ids
    )
    assert (
        "docker-airflow-preprocess-valid"
        in predict_dag_tasks["docker-airflow-predict"].upstream_task_ids
    )
