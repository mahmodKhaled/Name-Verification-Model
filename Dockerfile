FROM python:3.8-slim-buster
WORKDIR /home/src
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 80
COPY src/ /home/src
CMD ["uvicorn", "server_app:app","--host", "0.0.0.0", "--port", "80"]