import os
import json
from tqdm import tqdm
# requires pymupdf==1.22.5
import fitz
from haystack import Document
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from sentence_transformers import SentenceTransformer
import numpy as np
import umap.umap_ as umap
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer

class DocumentIngestion:
    def __init__(self, path, model, device = 'cpu', max_docs = float("inf"), n_topics = 10):
        self.path = path
        self.documents = []
        self.max_docs = max_docs
        self.device = device
        self.n_topics = n_topics
        print(path, device)

        if path.endswith('.pdf'):
            # Handle single PDF file
            self.ingest_pdf(path)
        elif path.endswith('.json'):
            # Handle JSON file
            self.load_json(path)
        elif os.path.isdir(path):
            self.ingest_pdf_folder(path)
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
        
        # Perform topic modeling with BERTopic
        self.create_topics()

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
                self.documents.append(Document(id = file_name, 
                                            content= content[0:4000],
                                            embedding = None,
                                            meta = {'embedding_2d': None, 'title': None}
                                            ))
        except:
            file_name = os.path.basename(pdf_path)
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

    def create_embeddings(self, model):
        model = SentenceTransformer(model, device = self.device)
   
        content = []
        for doc in tqdm(self.documents):
            content.append(doc.content)
        embeddings = model.encode(content, batch_size = 128, show_progress_bar = True)
        for i, doc in enumerate(tqdm(self.documents)):
            doc.embedding = embeddings[i].tolist()

        #clean up when done
        del embeddings

    def create_2d_embeddings(self):

        embeddings_np = np.array([doc.embedding for doc in self.documents])
        if len(embeddings_np) > 30000:
            # Sample 20,000 data points
            sampled_indices = np.random.choice(embeddings_np.shape[0], 30000, replace=False)
            sampled_data = embeddings_np[sampled_indices]
        else:
            sampled_data = embeddings_np
        # Train UMAP against the sampled data
        umap_model = umap.UMAP()
        print('Training 2d Representations...')
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
    
    def create_topics(self):
        # Convert the list of document embeddings into a numpy array
        embeddings_np = np.array([doc.embedding for doc in self.documents])
        print('Fitting BERTopic Model...')

        # Custom CountVectorizer to remove stop words
        vectorizer_model = CountVectorizer(ngram_range=(1, 3), stop_words='english')

        # Initialize the BERTopic model
        model = BERTopic(vectorizer_model=vectorizer_model)
        # model = BERTopic(n_gram_range=(1, 3), min_topic_size=10, top_n_words=10)

        # Fit the model on document contents with the precomputed embeddings
        topics, probs = model.fit_transform([doc.content for doc in self.documents], embeddings_np)

        # Extract top 5 words per topic
        topic_info = model.get_topic_info()
        top_words_per_topic = {}

        for topic_id in topic_info['Topic']:
            if topic_id != -1:  # Ignore outliers
                words = model.get_topic(topic_id)
                top_words_per_topic[topic_id] = [word for word, _ in words[:5]]

        # Store the generated topics and their probabilities into each document's metadata
        for i, doc in enumerate(self.documents):
            doc.meta['topic'] = topics[i]  # Assign the identified topic to the document
            doc.meta['topic_probability'] = probs[i]  # Assign the probability of the document belonging to the topic

        # Store the fitted BERTopic model & top words in the instance for potential future use
        self.topic_model = model
        self.top_words_per_topic = top_words_per_topic
