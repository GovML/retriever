from haystack import Pipeline, Document, component

from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever
from haystack.components.rankers import TransformersSimilarityRanker
from haystack.components.readers import ExtractiveReader
from haystack.components.joiners.document_joiner import DocumentJoiner

from transformers import pipeline

import gc
from typing import List, Optional

@component
class BertQA:
  """
  A component generating personal welcome message and making it upper case
  """
  @component.output_types(qa_answer=str)
  def run(self, query: str, documents: List[Document], top_k: Optional[int] = None):
    question_answerer = pipeline("question-answering", model="deepset/tinyroberta-squad2")
    c= ' '.join([d.content for d in documents])
    res = question_answerer(question=query, context=c)
    print(res)
    return {"qa_answer": res['answer'], "score": res['score']}

class QueryPipeline:
    def __init__(self, document_store, type, ann_model, ranker_model):
        self.document_store = document_store
        self.type = type
        self.ann_model = ann_model
        
        self.pipe = Pipeline()
        self.pipe.add_component("text_embedder", SentenceTransformersTextEmbedder(model=self.ann_model))
        self.pipe.add_component("retriever", QdrantEmbeddingRetriever(document_store=document_store, return_embedding= False))
        self.pipe.add_component("reranker", TransformersSimilarityRanker(ranker_model))
        self.pipe.add_component("qa", BertQA())
        self.pipe.add_component("search", DocumentJoiner())

        self.pipe.connect("text_embedder.embedding", "retriever.query_embedding")
        self.pipe.connect("retriever.documents", "reranker.documents")
        self.pipe.connect("retriever.documents", "search.documents")
        self.pipe.connect("reranker.documents", "qa.documents")

    def run(self, query):
        res = self.pipe.run(data = {"text_embedder": {"text": query},
                              "retriever": {'top_k': 20},
                              "reranker": {"query": query, "top_k": 3},
                              "qa": {"query": query, "top_k": 1},
                              })
        gc.collect()
        return res
