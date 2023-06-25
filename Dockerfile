FROM python:3.8-slim-buster

WORKDIR /home/src

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

EXPOSE 80

COPY src/ /home/src
COPY models/ /home/src/models

ENV FLASK_APP=server_app.py
CMD ["flask", "run", "--host=0.0.0.0", "--port=80"]
