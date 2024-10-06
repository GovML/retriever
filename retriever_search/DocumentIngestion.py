import os
import json
from tqdm import tqdm
# requires pymupdf==1.22.5
import fitz
import pandas as pd
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from sentence_transformers import SentenceTransformer
import numpy as np
import umap.umap_ as umap

class DocumentIngestion:
    def __init__(self, path, model, device = 'cpu', max_docs = float("inf")):
        self.path = path
        self.documents = []
        self.max_docs = max_docs
        self.device = device
        print(path, device)

        if path.endswith('.pdf'):
            # Handle single PDF file
            self.ingest_pdf(path)
        elif path.endswith('.json'):
            # Handle JSON file
            self.load_json(path)
        elif os.path.isdir(path):
            pdf = True
            for fname in os.listdir(path):
                if not fname.endswith('.pdf'):
                    pdf = False
                    break
            if pdf:
                self.ingest_pdf_folder(path)

            if not pdf:
                csv = True
                for fname in os.listdir(path):
                    if not fname.endswith('.csv'):
                        csv = False
                        break
                if csv:
                    print('csv files exist')
                    self.ingest_csv_folder(path)

        else:
            assert False, 'Unsupported file type(s) for ingestion. Please provide a folder or json.'

        #compute embeddings if not present
        if self.documents[0].embedding is None:
            print('Computing Embeddings...')
            self.create_embeddings(model = model)
        else:
            print('Embeddings loaded from json.')

        #compute 2d embeddings
        if self.documents[0].meta['embedding_2d'] is None:
            print('Computing 2d Embeddings...')

            self.create_2d_embeddings()
        else:
            print('2d Embeddings loaded from json.')

    def convert_pdf_str(self, pdf_path:str) -> str:
        #performant conversion of pdf path to string
        pdf_text = ''
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text = page.get_text()
                pdf_text += text + '\n'

        return pdf_text

    def ingest_pdf(self, pdf_path):
        # Convert single PDF to dictionary and add it to self.documents
        try:
            file_name = os.path.basename(pdf_path)
            #if len(self.documents) > 280:
            #    print(file_name)
            content = self.convert_pdf_str(pdf_path)
            if len(self.documents) >= self.max_docs:
                print('Hit max documents allowed by your instantiation of max_docs. No more documents to be added.')
                return
            else:
                # first 4000 tokens (chunking)
                self.documents.append(Document(id = file_name, 
                                            content= content[0:4000],
                                            embedding = None,
                                            meta = {'embedding_2d': None, 'title': None}
                                            ))
                print('documents appended')
        except:
            file_name = os.path.basename(pdf_path)
            print(file_name)

    def convert_csv_str(self, csv_path: str) -> str:
        # performant conversion of pdf path to string
        csv_text = ''
        df = pd.read_csv(csv_path)
        text_series_list = [df[col].astype(str) for col in df.columns]
        text_strings = [' '.join(text_series) for text_series in text_series_list]
        for text_string in text_strings:
            csv_text += text_string
        return csv_text

    def ingest_csv(self, csv_path):
        try:
            file_name = os.path.basename(csv_path)
            content = self.convert_csv_str(csv_path)
            print('content: ', content)
            if len(self.documents) >= self.max_docs:
                print('Hit max documents allowed by your instantiation of max_docs. No more documents to be added.')
                return
            else:
                self.documents.append(Document(id = file_name,
                                            content= content[0:4000],
                                            embedding = None,
                                            meta = {'embedding_2d': None, 'title': None}
                                            ))
                print('documents appended')
        except:
            file_name = os.path.basename(csv_path)
            print(file_name)
        

    def load_json(self, json_path):
        # Load JSON file and extract documents\
        print(f'Loading from JSON: {json_path}')

        with open(json_path, 'r') as json_file:
            data = json.load(json_file)

            for item in tqdm(data):
                if 'embedding' in item.keys():
                    emb = item["embedding"]
                else:
                    emb = None
                if 'embedding_2d' in item.keys():
                    emb_2d = item["embedding_2d"]
                else:
                    emb_2d  = None

                if 'title' in item.keys():
                    title= item["title"]
                else:
                    title= None

                self.documents.append(Document(id = item["id"], 
                                               content= item["content"][0:4000],
                                               embedding = emb,
                                               meta = {'embedding_2d': emb_2d, 'title': title}
                                               ))
                if len(self.documents) >= self.max_docs:
                    print('Hit max documents allowed by your instantiation of max_docs. No more documents to be added.')
                    return


    def ingest_pdf_folder(self, path):
        # Convert all PDFs in a directory to documents
        for file_name in tqdm(os.listdir(self.path)):
            if file_name.endswith('.pdf'):
                pdf_path = os.path.join(self.path, file_name)
                self.ingest_pdf(pdf_path)

    def ingest_csv_folder(self, path):
        for file_name in tqdm(os.listdir(self.path)):
            if file_name.endswith('.csv'):
                csv_path = os.path.join(self.path, file_name)
                print('ingesting csv files')
                self.ingest_csv(csv_path)

    def create_embeddings(self, model):
        model = SentenceTransformer(model, device = self.device)

        content = []
        print("document1 ", self.documents)
        for doc in tqdm(self.documents):
            content.append(doc.content)
        embeddings = model.encode(content, batch_size = 128, show_progress_bar = True)
        print("embeddings ", embeddings)
        for i, doc in enumerate(tqdm(self.documents)):
            doc.embedding = embeddings[i].tolist()
        print("document2 ", self.documents)
        #clean up when done
        del embeddings

    def create_2d_embeddings(self):

        embeddings_np = np.array([doc.embedding for doc in self.documents])
        if len(embeddings_np) > 30000:
            # Sample 20,000 data points
            sampled_indices = np.random.choice(embeddings_np.shape[0], 30000, replace=False)
            sampled_data = embeddings_np[sampled_indices]
            print("sampled_data after ", sampled_data)
        else:
            sampled_data = embeddings_np
        # Train UMAP against the sampled data
        umap_model = umap.UMAP()
        print('Training 2d Representations...')

        print("sampled_data ", sampled_data)
        umap_model.fit(sampled_data)

        print('Fitting 2d Representations...')
        # Run UMAP against the entire embeddings_np
        embeddings_2d = umap_model.transform(embeddings_np)
        del embeddings_np
        del umap_model
        del sampled_data

        print('Inserting into Document List')
        for i, emb_2d in enumerate(tqdm(embeddings_2d)):
            self.documents[i].meta['embedding_2d'] = emb_2d.tolist()

    def to_json(self, output_path='output.json'):
        # Convert self.documents to JSON
        print('Saving documents to json...')
        dic = []
        for d in tqdm(self.documents):
            dic.append(d.to_dict())

        print('Creating json file...')
        with open(output_path, 'w') as json_file:
            json.dump(dic, json_file)

        del dic
        print(f'Json saved to {output_path}. Initiate search with the json to not repeat ingestion.')
