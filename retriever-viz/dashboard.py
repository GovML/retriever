import requests
import pandas as pd
import ast
import dash
from dash import Dash, html, dcc, callback, Output, Input
from flask import Flask
import io
import base64

#plotting utils
from wordcloud import WordCloud
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.feature_extraction import text

from haystack import Document

class Dashboard():
    def __init__(self):
        self.documents = {}
        self.update_documents()
        # set up server
        self.server = Flask(__name__)
        self.app = dash.Dash(__name__, server=self.server)

        # Initialize app layout
        self.app.layout = html.Div([
            dcc.Dropdown(list(self.documents.keys()), list(self.documents.keys()), id='query-selector', multi=True),
            # dcc.Graph(id='graph-content'),
            # html.Img(id='wordcloud'),
            dcc.Graph(id='heatmap')
        ])

        self.app.callback(
            Output('heatmap', 'figure'),
            Input('query-selector', 'value')
            )(self.update_graphs)
        
    def update_documents(self):
        search_results = requests.get("http://127.0.0.1:5000/search_results").json()
        embeddings_results = requests.get("http://127.0.0.1:5000/search_return_embeddings").json()

        for i, result in enumerate(search_results):
            embeddings = embeddings_results[i]['embedding_2d'] #Life Beyond the Solar System Space Weather and Its Impact on Habitable Worlds.pdf
            query = result['query']

            if query not in self.documents.keys():
                self.documents[query] = []
                data = ast.literal_eval(result['data'])
                
                for z, doc in enumerate(data):
                    embedding = embeddings[z]['embeddings']
                    self.documents[query].append(
                        Document(id = doc['Link'], 
                            content= doc['Content'],
                            embedding = None,
                            meta = {'embedding_2d': embedding, 'title': doc['Title']}
                            )
                    )

    #runs the app from main
    def run(self):
        self.app.run_server(debug=True, host="127.0.0.1", port=8055)


    ##### Visualization Code Start

    #creates heatmap
    def update_heatmap_graph(self, selected_queries):

        texts = []
        for query, query_documents in self.documents.items():
            if query in selected_queries: #checks if is one of selected queries
                for doc in query_documents:
                    texts.append(doc.content)

        stop_words = list(text.ENGLISH_STOP_WORDS.union(['additional', 'stopwords', 'if', 'needed']))
        vectorizer = text.CountVectorizer(max_features=50, stop_words=stop_words)
        X = vectorizer.fit_transform(texts)
        df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        fig = px.imshow(df, labels=dict(x="Words", y="Documents", color="Frequency"),
                        x=df.columns, y=["Doc {}".format(i) for i in range(len(texts))])
        fig.update_layout(title="Word Frequency Heatmap")
        return fig
    
    ##### Visualization Code End


    def update_graphs(self, selected_queries):
        heatmap_fig = self.update_heatmap_graph(selected_queries)
        return heatmap_fig


if __name__ == "__main__":
    dash_app = Dashboard()
    dash_app.run()