import requests
import pandas as pd
import ast
import dash
from dash import Dash, html, dcc, callback, Output, Input
from flask import Flask
import io
import base64
import pycountry
from geopy.geocoders import Nominatim
from collections import Counter


#plotting utils
from wordcloud import WordCloud
import plotly.express as px
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('agg')
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
            html.Button('Refresh Data', id='refresh-button', n_clicks=0),
            dcc.Graph(id='umap'),
            dcc.Graph(id='heatmap'),
            dcc.Graph(id='location-map'),
            html.Img(id='wordcloud'),
        ])

        self.app.callback(
            Output('umap','figure'),
            Output('heatmap', 'figure'),
            Output('location-map', 'figure'),
            Output('wordcloud','src'),
            
            Input('query-selector', 'value')
            )(self.update_graphs)
        
        self.app.callback(
            Output('query-selector', 'options'),
            Output('query-selector', 'value'),
            Input('refresh-button', 'n_clicks')
        )(self.refresh_data)

    def update_documents(self):
        search_results = requests.get("http://127.0.0.1:5000/get_all_results_cache").json()
        self.documents = search_results

    def refresh_data(self, n_clicks):
        self.update_documents()
        return list(self.documents.keys()), []

    #runs the app from main
    def run(self):
        self.app.run_server(debug=True, host="127.0.0.1", port=8055)


    ##### Visualization Code Start
    def update_2d_graph(self, selected_queries):
        embeddings = []
        titles = []
        for query, query_documents in self.documents.items():
            if query in selected_queries: #checks if is one of selected queries
                for doc in query_documents:
                    embeddings.append(doc['meta']['embedding_2d'])
                    titles.append(doc['meta']['title'])

        emb_df = pd.DataFrame(embeddings, columns = ['x','y'])
        emb_df['Title'] = titles   
        fig = px.scatter(emb_df, x='x', y='y', custom_data=['Title'])
        fig.update_traces(
        hovertemplate="<br>".join([
            "%{customdata[0]}",
        ])
        )
        fig.update_layout(transition=dict(duration=500, easing='cubic-in-out'))

        return fig

    #creates heatmap
    def update_heatmap_graph(self, selected_queries):
        texts = []
        for query, query_documents in self.documents.items():
            if query in selected_queries: #checks if is one of selected queries
                for doc in query_documents:
                    texts.append(doc['content'])

        stop_words = list(text.ENGLISH_STOP_WORDS.union(['additional', 'stopwords', 'if', 'needed']))
        [stop_words.append(str(i)) for i in range(100)]
        vectorizer = text.CountVectorizer(max_features=100, stop_words=stop_words)
        X = vectorizer.fit_transform(texts)
        df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        fig = px.imshow(df, labels=dict(x="Words", y="Documents", color="Frequency"),
                        x=df.columns, y=["Doc {}".format(i) for i in range(len(texts))])
        fig.update_layout(title="Word Frequency Heatmap")
        return fig
    
    #makes wordcloud
    def update_wordcloud_graph(self, selected_queries):
        texts = []
        for query, query_documents in self.documents.items():
            if query in selected_queries: #checks if is one of selected queries
                for doc in query_documents:
                    texts.append(doc['content'])
        
        texts = ' '.join(texts).split()

        stop_words = list(text.ENGLISH_STOP_WORDS.union(['additional', 'stopwords', 'if', 'needed']))        
        filtered_text = [word for word in texts if word.lower() not in stop_words and len(word) > 2]

        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(filtered_text)).to_image()

        img = io.BytesIO()
        wordcloud.save(img, format='PNG')
        return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())
    


    # method to extract locations from text
    def extract_locations(self, text):
        countries = {country.name: country.alpha_3 for country in pycountry.countries}
        words = text.split()
        locations = [countries[word] for word in words if word in countries]
        return locations
    
    # creates location map
    def update_location_map(self, selected_queries):
        texts = []
        for query, query_documents in self.documents.items():
            if query in selected_queries:  # checks if is one of selected queries
                for doc in query_documents:
                    texts.append(doc['content'])

        location_counts = Counter()
        for text in texts:
            location_counts.update(self.extract_locations(text))

        df = pd.DataFrame(location_counts.items(), columns=['country_code', 'count'])
        df['country'] = df['country_code'].apply(lambda x: pycountry.countries.get(alpha_3=x).name)

        fig = px.choropleth(df, locations='country_code', color='count',
                            hover_name='country', color_continuous_scale='Viridis', projection='natural earth',
                            title='Country Mentions')

        fig.update_geos(showcoastlines=True, coastlinecolor="Black",
                        showland=True, landcolor="white",
                        showocean=True, oceancolor="lightblue",
                        showlakes=True, lakecolor="lightblue",
                        showrivers=True, rivercolor="lightblue")

        return fig
    ##### Visualization Code End

    def update_graphs(self, selected_queries):
        
        if len(selected_queries) !=0:
            umap_fig = self.update_2d_graph(selected_queries)
            heatmap_fig = self.update_heatmap_graph(selected_queries)
            location_map_fig = self.update_location_map(selected_queries)

            wordcloud_src = self.update_wordcloud_graph(selected_queries)
            return umap_fig, heatmap_fig, location_map_fig, wordcloud_src

        
        else:
            return {}, {}, {}, ''
            


def run_viz_server():
    dash_app = Dashboard()
    dash_app.run()