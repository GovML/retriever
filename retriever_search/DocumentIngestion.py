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
    def __init__(self, path, model, ingestion_type= 'document', device = 'cpu', max_docs = float("inf"), n_topics = 10):
        """
        Args:
            path (str): Path to the file or folder to ingest.
            model (str): Model to compute embeddings.
            device (str): Device for computation ('cpu' or 'cuda').
            max_docs (int): Maximum number of documents to ingest.
            n_topics (int): Number of topics for topic modeling.
            ingestion_type (str): "document" for full-document ingestion, "page" for page-wise ingestion.
        """
        assert ingestion_type in {"document", "page"}, 'ingestion_type must be either "document" or "page".'

        self.path = path
        self.documents = []
        self.max_docs = max_docs
        self.device = device
        self.n_topics = n_topics
        self.ingestion_type = ingestion_type
        print(path, device, ingestion_type)

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
        try:
            file_name = os.path.basename(pdf_path)
            with fitz.open(pdf_path) as doc:
                current_content = ""
                start_page = 1  # Tracks the starting page of a merged document

                for page_num, page in enumerate(doc):
                    page_content = page.get_text().strip()

                    if not current_content:
                        start_page = page_num + 1  # Initialize start_page

                    # Add current page content to current_content
                    current_content += page_content

                    # Check if we need to merge the next page or create a new document
                    if len(current_content) >= 1000 or page_num == len(doc) - 1:  # Either content is sufficient or last page
                        # Create a document with the accumulated content
                        if len(self.documents) >= self.max_docs:
                            print('Hit max documents allowed by your instantiation of max_docs. No more documents to be added.')
                            return

                        # Define page range in the ID
                        end_page = page_num + 1
                        page_range = f"{start_page}" if start_page == end_page else f"{start_page}-{end_page}"
                        self.documents.append(Document(
                            id=f"{file_name}_page_{page_range}",
                            content=current_content,
                            embedding=None,
                            meta={'embedding_2d': None, 'title': file_name}
                        ))

                        # Reset current_content for the next document
                        current_content = ""
        except Exception as e:
            print(f"Failed to process PDF {pdf_path}: {e}")

        

    def load_json(self, json_path):
        print(f'Loading from JSON: {json_path}')
        with open(json_path, 'r') as json_file:
            data = json.load(json_file)

            for item in tqdm(data):
                self.documents.append(Document(
                    id=item["id"],
                    content=item["content"][0:4000],
                    embedding=item.get("embedding"),
                    meta={'embedding_2d': item.get("embedding_2d"), 'title': item.get("title")}
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
