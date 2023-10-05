from fastapi import FastAPI
from pydantic import BaseModel

from chromadb.utils import embedding_functions

app = FastAPI()
embedding_function = embedding_functions.DefaultEmbeddingFunction()


class EmbeddingRequest(BaseModel):
    value: str


class EmbeddingResponse(BaseModel):
    value: list[float]


@app.get("/")
async def root():
    return {"message": "OK"}


@app.post("/embedding")
async def embedding(request: EmbeddingRequest) -> EmbeddingResponse:
    em = embedding_function([request.value])
    return EmbeddingResponse(value=em[0])
