FROM python:3.7-slim-stretch

COPY train.py /train.py
COPY requirements.txt /requirements.txt
RUN pip3 install --upgrade pip && pip3 install -r requirements.txt

ENTRYPOINT ["python", "train.py"]
