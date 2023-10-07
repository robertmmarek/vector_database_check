FROM python:3.9-slim-bullseye

WORKDIR /app

COPY ./src ./src

RUN apt update

RUN yes | apt install build-essential

RUN pip install qdrant-client==1.5.4

RUN pip install pysqlite3-binary

RUN pip install chromadb==0.4.1

RUN pip install fastapi==0.103.2

RUN pip install onnxruntime==1.15.1

RUN pip install "uvicorn[standard]"

WORKDIR /app/src

CMD ["uvicorn", "server:app", "--port", "6433"]