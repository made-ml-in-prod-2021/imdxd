FROM python:3.7-slim-stretch

COPY upload_data.py /upload_data.py
COPY requirements.txt /requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

ENTRYPOINT ["python", "upload_data.py"]
