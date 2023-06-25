FROM python:3.8-slim-buster
WORKDIR /home/src
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 80
COPY src/ /home/src
COPY models/ /home/src/models
CMD ["python", "server_app.py"]