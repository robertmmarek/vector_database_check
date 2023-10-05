FROM python:3.12.0rc2-alpine3.18

WORKDIR /app

COPY ./src ./src

RUN pip install qdrant-client

CMD ["python", "python_console.py"]