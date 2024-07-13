import sys
import os
import pandas as pd
from flask import Flask, jsonify, request


from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from retriever_search.DocumentIngestion import DocumentIngestion
from retriever_search.QueryPipeline import QueryPipeline

class SearchServer:

    def __init__(self, input_directory = None, input_json = None, json_save_path = None, embedding_model = 'sentence-transformers/allenai-specter', device = 'cpu', host = '127.0.0.1', verbose = True):
        if verbose:
            print('Initializing Application...')

        self.app = Flask(__name__)
        self.setup_routes()
        self.embedding_model = embedding_model
        '''
        other options:
        'sentence-transformers/all-mpnet-base-v2'
        'sentence-transformers/allenai-specter'
        'Dagar/t5-small-science-papers'
        '''
        self.device = device
        self.host = host

        self.query_result_cache = {}
        self.search_result_cache = {}

        #check input logic
        if input_json is not None:
            if verbose:
                print('Initializing from JSON as preferred ingestion method.')
            self.input_file = input_json

        elif input_directory is not None:
            if verbose:
                print('Initializing from input directory')
            self.input_file = input_directory
        
        else:
            assert False, 'Please provide input_directory or json_read_path to initialize the server.'

        if json_save_path is None and input_json is None:
            print('WARNING: you have not saved your outputs and will need to re-ingest your data. We recommend passing in a value like "myjson.json" for json_save_path to store your ingested documents.')

        if verbose:
            print('Initializing Document Ingestion...')
        #inputs should be clear by this point

        ingestion = DocumentIngestion(self.input_file, model = self.embedding_model, device=self.device)

        if json_save_path is not None:
            ingestion.to_json(json_save_path)
        
        self.document_store = document_store = QdrantDocumentStore(
            ":memory:",
            embedding_dim=768,  # the embedding_dim should match that of the embedding model
        )

        self.document_store.write_documents(ingestion.documents)

        if verbose:
            print('Initializing Search Functions...')

        self.query_engine = QueryPipeline(document_store, 
                            type = 'Search', 
                            ann_model=self.embedding_model, 
                            ranker_model="cross-encoder/ms-marco-MiniLM-L-12-v2"
                            )
        if verbose:
            print('Initializing Document Ingestion...')

        self.app.run(debug=True, host='127.0.0.1', threaded=True, port = 5000)
    

    def setup_routes(self):
        '''
        all flask app routes go here
        '''
        @self.app.route('/search_query', methods=['POST'])
        def search_query():
            query = request.get_json()['query']
            if query not in self.query_result_cache.keys():
                responses = self.query_engine.run(query=str(query))
                self.query_result_cache[query] = responses
                self.search_result_cache[query] = responses['search']['documents']
            else:
                responses = self.query_result_cache[query]
            return jsonify(responses)
        
        @self.app.route('/get_previous_queries', methods=['GET'])
        def get_previous_queries():
            return jsonify(list(self.query_result_cache.keys()))  
        
        @self.app.route('/get_all_results_cache', methods=['GET'])
        def get_all_results_cache():
            return jsonify(self.search_result_cache)

def run_search_server(input_directory = None, input_json = None, json_save_path = None, embedding_model = 'sentence-transformers/allenai-specter', device = 'cpu', host = '127.0.0.1', verbose = True):
    SearchServer(input_directory,input_json,json_save_path,embedding_model,device,host,verbose) #, json_save_path = '../testing_new_server.json',device = 'mps'