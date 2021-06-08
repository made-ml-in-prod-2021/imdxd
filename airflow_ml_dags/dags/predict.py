from datetime import timedelta

from airflow import DAG
from airflow.sensors.filesystem import FileSensor
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "predict",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(7),
) as dag:
    wait_data = FileSensor(
        task_id="wait_for_prediction_data",
        filepath="/opt/airflow/data/raw/{{ ds }}/data.csv",
        poke_interval=30,
    )

    wait_model = FileSensor(
        task_id="wait_for_prediction_model",
        filepath="/opt/airflow/data/models/{{ var.value.model }}/model.pkl",
        poke_interval=30,
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input_dir /data/raw/{{ ds }} --output_dir /data/proceed/{{ ds }}",
        task_id="docker-airflow-preprocess-valid",
        do_xcom_push=False,
        volumes=["/home/imd/Projects/made_ml_prod/airflow_ml_dags/data:/data"],
    )

    predict = DockerOperator(
        image="airflow-predict",
        command="--input_dir /data/proceed/{{ ds }} --prediction_dir /data/predictions/{{ ds }} "
        "--model_dir /data/models/{{ var.value.model }}",
        task_id="docker-airflow-predict",
        do_xcom_push=False,
        volumes=["/home/imd/Projects/made_ml_prod/airflow_ml_dags/data:/data"],
    )

    [wait_data, wait_model] >> preprocess >> predict
