FROM python:3.9-slim-bullseye

WORKDIR /app

COPY ./src ./src

RUN apt update

RUN yes | apt install build-essential

RUN pip install chromadb==0.4.1

CMD ["python", "server.py"]