from datetime import timedelta

from airflow import DAG
from airflow.models import Variable
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sensors.filesystem import FileSensor
from airflow.utils.dates import days_ago

OUTPUT_DIR = Variable.get("OUTPUT_DIR")

default_args = {
    "owner": "imd",
    "email": ["imdxdd@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "train",
    default_args=default_args,
    schedule_interval="@weekly",
    start_date=days_ago(7),
) as dag:

    wait_data = FileSensor(
        task_id="wait_for_train_data",
        filepath="/opt/airflow/data/raw/{{ ds }}/data.csv",
        poke_interval=30,
    )

    wait_target = FileSensor(
        task_id="wait_for_train_target",
        filepath="/opt/airflow/data/raw/{{ ds }}/target.csv",
        poke_interval=30,
    )

    preprocess = DockerOperator(
        image="airflow-preprocess",
        command="--input_dir /data/raw/{{ ds }} --output_dir /data/proceed/{{ ds }}",
        task_id="docker-airflow-preprocess-train",
        do_xcom_push=False,
        volumes=[f"{OUTPUT_DIR}:/data"],
    )

    split = DockerOperator(
        image="airflow-split",
        command="--input_dir /data/proceed/{{ ds }} --output_dir /data/experiment/{{ ds }}",
        task_id="docker-airflow-split",
        do_xcom_push=False,
        volumes=[f"{OUTPUT_DIR}:/data"],
    )

    train = DockerOperator(
        image="airflow-train",
        command="--input_dir /data/experiment/{{ ds }} --model_dir /data/models/{{ ds }}",
        task_id="docker-airflow-train",
        do_xcom_push=False,
        volumes=[f"{OUTPUT_DIR}:/data"],
    )

    validate = DockerOperator(
        image="airflow-validate",
        command="--input_dir /data/experiment/{{ ds }} --model_dir /data/models/{{ ds }} "
        "--metric_dir /data/metrics/{{ ds }}",
        task_id="docker-airflow-validate",
        do_xcom_push=False,
        volumes=[f"{OUTPUT_DIR}:/data"],
    )

    [wait_data, wait_target] >> preprocess >> split >> train >> validate
