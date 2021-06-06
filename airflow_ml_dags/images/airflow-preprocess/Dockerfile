FROM python:3.7-slim-stretch

COPY preprocess_data.py /preprocess_data.py
COPY requirements.txt /requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

ENTRYPOINT ["python", "preprocess_data.py"]
