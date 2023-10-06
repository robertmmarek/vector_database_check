import os

from fastapi import FastAPI
from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from chromadb.utils import embedding_functions

QDRANT_HOST = os.environ["QDRANT_SERVER_HOSTNAME"]
QDRANT_PORT = os.environ["QDRANT_SERVER_PORT"]

COLLECTION_NAME = "embedding_collection"

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
    q_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
    em = embedding_function([request.value])[0]

    embedding_size = len(em)

    collection = q_client.get_collection(collection_name=COLLECTION_NAME)
    if not collection:
        q_client.recreate_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=embedding_size, distance=Distance.DOT),
        )

    vector_exist_check = q_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=em,
        query_filter=Filter(
            must=[FieldCondition(key="text", match=MatchValue(value=request.value))]
        ),
        limit=1,
    )

    if not vector_exist_check:
        insert_res = q_client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=[PointStruct(vector=em, payload={"text": request.value})],
        )

        print(insert_res)

    return EmbeddingResponse(value=em)
