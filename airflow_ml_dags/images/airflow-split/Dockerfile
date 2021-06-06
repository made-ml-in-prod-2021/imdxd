FROM python:3.7-slim-stretch

COPY split_data.py /split_data.py
COPY requirements.txt /requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

ENTRYPOINT ["python", "split_data.py"]
