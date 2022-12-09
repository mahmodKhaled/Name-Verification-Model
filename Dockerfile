FROM python:3.8-slim-buster
WORKDIR /home/app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 80
COPY app/ /home/app
CMD ["uvicorn", "server_app:app","--host", "0.0.0.0", "--port", "80"]