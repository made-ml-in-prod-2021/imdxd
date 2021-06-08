from datetime import timedelta

from airflow import DAG
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.utils.dates import days_ago

default_args = {
    "owner": "imd",
    "email": ["imdxdd@gmail.com"],
    "email_on_failure": True,
    "email_on_retry": True,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}


with DAG(
    "upload",
    default_args=default_args,
    schedule_interval="@daily",
    start_date=days_ago(1),
) as dag:

    download = DockerOperator(
        image="airflow-upload",
        command="/data/raw/{{ ds }}",
        network_mode="bridge",
        task_id="docker-airflow-upload",
        do_xcom_push=False,
        volumes=["/home/imd/Projects/made_ml_prod/airflow_ml_dags/data:/data"],
    )
