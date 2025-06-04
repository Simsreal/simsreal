# import time

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
import logging

logger = logging.getLogger(__name__)


class VectorStore:
    # emotion (vec), emotion (cat.), timestamp
    """
    stores latent with associated emotions and timestamps

    add:
        vec_store.add(
            collection_name="sample_collection",
            documents=["sample document"],
            metadata=[{"source": "sample_source"}],
            ids=[1]
        )

    query:
        vec_store.query(
            collection_name="sample_collection",
            query_text="sample query"
        )
    """

    def __init__(
        self,
        host,
        port,
        reset,
        vector_size,
        collection_name,
        gpu=False,
        create=True,
    ):
        self.collection_name = collection_name

        self.client = QdrantClient(
            host=host,
            port=port,
        )

        if gpu:
            # Try CUDA first, fallback to CPU if not available
            try:
                self.client.set_model(
                    self.client.DEFAULT_EMBEDDING_MODEL,
                    providers=[
                        "CUDAExecutionProvider",
                        "CPUExecutionProvider",
                    ],
                )
            except ValueError as e:
                if "CUDAExecutionProvider is not available" in str(e):
                    logger.warning("CUDA not available for ONNX runtime, falling back to CPU")
                    self.client.set_model(
                        self.client.DEFAULT_EMBEDDING_MODEL,
                        providers=["CPUExecutionProvider"],
                    )
                else:
                    raise e

        if reset and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)

        if create and not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
