FROM python:3.6-slim-stretch

WORKDIR /app
COPY ./src /app/src
COPY ./webserver.py /app/webserver.py
COPY ./model.pkl /app/model.pkl
COPY ./requirements.txt /app/requirements.txt

RUN mkdir /app/logs
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

EXPOSE 4000/tcp

CMD ["gunicorn", "-b 0.0.0.0:4000", "-w 4", "webserver:app"]
