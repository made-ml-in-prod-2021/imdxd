from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "airflow",
    "email": ["airflow@example.com"],
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "train",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=days_ago(7),
) as dag:

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input_dir /data/raw/{{ ds }} --output_dir /data/proceed/{{ ds }}",
        task_id="docker-airflow-preprocess",
        do_xcom_push=False,
        volumes=["/home/imd/Projects/made_ml_prod/airflow_ml_dags/data:/data"],
    )

    split = DockerOperator(
        image="airflow-split",
        command="--input_dir /data/proceed/{{ ds }} --output_dir /data/experiment/{{ ds }}",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        volumes=["/home/imd/Projects/made_ml_prod/airflow_ml_dags/data:/data"],
    )

    train = DockerOperator(
        image="airflow-train",
        command="--input_dir /data/experiment/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        volumes=[
            "/home/imd/Projects/made_ml_prod/airflow_ml_dags/data:/data",
            "/home/imd/Projects/made_ml_prod/airflow_ml_dags/mlruns:/mlruns"
        ],
    )

    preprocess >> split >> train
