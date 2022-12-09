FROM python:3.11-buster
WORKDIR /home/app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
EXPOSE 5000
COPY app/ /home/app
CMD ["uvicorn", "server_app:app --reload"]