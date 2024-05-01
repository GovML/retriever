import sys
import os
from retriever_search.DocumentIngestion import DocumentIngestion
from retriever_search.QueryPipeline import QueryPipeline
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore


class search_app:
    def __init__(self, inputDirectory, json_save_path, embedding_2d = "red_emb_1K.npy", device = 'cpu', model = 'sentence-transformers/allenai-specter'):
        self.embedding_2d = embedding_2d
        self.json_path = json_save_path
        self.inputDirectory = inputDirectory
        self.device = device
        self.model = model

    def new_search_run(self):
        #model = 'sentence-transformers/all-mpnet-base-v2'
        #model = 'sentence-transformers/allenai-specter'
        #model = 'Dagar/t5-small-science-papers'
        if os.path.exists(self.json_path):
            ingestion = DocumentIngestion(self.json_path, model = None, device=self.device)
        else:
            ingestion = DocumentIngestion(self.inputDirectory, model = self.model, device=self.device)
            ingestion.to_json(self.json_path)

        document_store = document_store = QdrantDocumentStore(
            ":memory:",
            embedding_dim=768,  # the embedding_dim should match that of the embedding model
        )

        document_store.write_documents(ingestion.documents)

        pipe = QueryPipeline(document_store, 
                            type = 'Search', 
                            ann_model=self.model, 
                            ranker_model="cross-encoder/ms-marco-MiniLM-L-12-v2"
                            )
        return pipe
