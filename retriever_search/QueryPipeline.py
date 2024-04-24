from haystack import Pipeline
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.readers import ExtractiveReader
from haystack.components.joiners.document_joiner import DocumentJoiner
import gc

class QueryPipeline:
    def __init__(self, document_store, type, ann_model, ranker_model):
        self.document_store = document_store
        self.type = type
        self.ann_model = ann_model
        
        self.pipe = Pipeline()
        self.pipe.add_component("text_embedder", SentenceTransformersTextEmbedder(model=self.ann_model))
        self.pipe.add_component("retriever", QdrantEmbeddingRetriever(document_store=document_store, return_embedding= False))
        self.pipe.add_component("reranker", TransformersSimilarityRanker(ranker_model))
        self.pipe.add_component("qa", ExtractiveReader())
        self.pipe.add_component("search", DocumentJoiner())
        #self.pipe.add_component("retriever_search", DocumentJoiner())

        self.pipe.connect("text_embedder.embedding", "retriever.query_embedding")
        self.pipe.connect("retriever.documents", "reranker.documents")
        self.pipe.connect("retriever.documents", "search.documents")
        self.pipe.connect("reranker.documents", "qa.documents")
        #self.pipe.connect("reranker.documents", "search.documents")
        # self.pipe.connect("reranker.documents", r)

    def run(self, query):
        res = self.pipe.run(data = {"text_embedder": {"text": query},
                              "retriever": {'top_k': 20},
                              "reranker": {"query": query, "top_k": 3},
                              "qa": {"query": query, "top_k": 1},
                              })
        gc.collect()
        return res
