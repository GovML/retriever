import requests
import pandas as pd
import ast
import dash
from dash import Dash, html, dcc, callback, Output, Input, ctx
from flask import Flask
import io
import base64
import json
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
        self.title_content_map = {}
        self.update_documents()
        self.header_style = {'color': "#1F618D", 'font-family': 'Roboto, sans-serif','margin-left': '0%'}
        self.button_style = {   'margin-left': '0%',
                                'margin-top' : '20px',
                                'font-size': '15px',
                                'padding': '5px 20px',
                                'text-align': 'center',
                                'text-decoration': 'none',
                                'cursor': 'pointer',
                                'border': '2px solid #3498db',
                                'border-radius': '5px',
                                'transition': 'background-color 0.3s, color 0.3s, border-color 0.3s',
                                'background-color': '#3498db',
                                'color': '#ffffff'
                            }
        # set up server
        self.server = Flask(__name__)
        self.app = dash.Dash(__name__, server=self.server)

        #figure trace updates
        

        # Initialize app layout
        self.app.layout = html.Div([
            html.H1("Analysis Dashboard", style = {'color': "#1F618D", 'font-family': 'Roboto, sans-serif','margin-left': '0%'}),
            dcc.Dropdown(list(self.documents.keys()), list(self.documents.keys()), id='query-selector', multi=True, style = {
                                                                                                            'display': 'block',
                                                                                                            'width': '80%',
                                                                                                            'margin-left': '0%',
                                                                                                            'background-color': '#F6FBFF',
                                                                                                            'border-color': '#DFE6E9',
                                                                                                            'font-family': 'Arial',
                                                                                                            'font-size': '18px',
                                                                                                        },),
            html.Button('Refresh Data', id='refresh-button', n_clicks=0, style = self.button_style),
            html.H2("2D Plot of the Related Papers", style = self.header_style),

            

            
            dcc.Graph(id='umap', style= {'margin-left' : '0%',"width": "85%",'padding': '0px 0px', 'box-shadow': '2px 2px 2px lightgrey'}),
            html.Br(),
            html.H2("Word Frequency Map", style = self.header_style),
            dcc.Graph(id='heatmap', style= {'margin-left' : '0%',"width": "85%",'padding': '0px 0px', 'box-shadow': '2px 2px 2px lightgrey'}),
            html.H2("Country Mentions", style = self.header_style),
            dcc.Graph(id='location-map', style= {'margin-left' : '0%',"width": "85%",'padding': '0px 0px', 'box-shadow': '2px 2px 2px lightgrey'}),
            html.H2("Word Cloud", style = self.header_style),
            html.Div([
            html.Img(id='wordcloud', disable_n_clicks = True, alt = "WordCloud", style = {'margin-top':'0px','height': '22.5%', 'width':'50%', 'box-shadow': '2px 2px 2px lightgrey', 'position': 'absolute'})], 
                style={'margin-left': '15%', 'margin-top': '0px', 'height': '40%', 'width': '`16%'}),
            
        ])


        self.app.callback(
            Output('umap','figure', allow_duplicate=True),
            Output('heatmap', 'figure',  allow_duplicate=True),
            Output('location-map', 'figure',  allow_duplicate=True),
            Output('wordcloud','src',  allow_duplicate=True),
            
            Input('query-selector', 'value'),
            prevent_initial_call=True
            )(self.update_graphs)
        
        self.app.callback(
            Output('heatmap', 'figure'),
            Output('location-map', 'figure'),
            Output('wordcloud','src'),
            Input('umap', 'selectedData'),
            prevent_initial_call = True,
        )(self.update_graphs)
        
        self.app.callback(
            Output('query-selector', 'options'),
            Output('query-selector', 'value'),
            Input('refresh-button', 'n_clicks')
        )(self.refresh_data)

        

    def fig_trace_update(self,fig, paper_len = 0):
            '''Perform basic updates to the figure
            Input: Figure object, Length of paper
            Output: enhanced figure object
            '''

            fig.update_traces(marker_size=6 )
            fig.update_xaxes(visible=False)
            fig.update_yaxes(visible=False)

            fig.update_layout(
                    showlegend=False,                    
                    autosize = True,
                    margin=dict(l= 5, r = 5, t=5, b=5),
                    plot_bgcolor="rgba(242,242,242, 0.5)",
                    transition={
                            'duration': 1000,  # 1 second
                            'easing': 'cubic-in-out'
                    }
                    )
            
            if paper_len == 0:
                fig.update_layout(template = "plotly_white")


            return fig

    def update_documents(self):
        search_results = requests.get("http://127.0.0.1:5000/get_all_results_cache").json()
        self.documents = search_results
        # with open("test.json", "r" ) as f:
        #     self.documents = json.load(f)
        
        for topic_doc in self.documents.values():
            for document in topic_doc:
                title = document["id"].replace(".pdf" , "")
                title = title.replace("_", " ")
                self.title_content_map[title] = document["content"]
        
    def refresh_data(self, n_clicks):
        self.update_documents()
        return list(self.documents.keys()), []

    #runs the app from main
    def run(self):
        self.app.run_server(debug=False, host="127.0.0.1", port=8055)


    ##### Visualization Code Start
    def update_2d_graph(self, selected_queries):
        embeddings = []
        titles = []
        for query, query_documents in self.documents.items():
            if query in selected_queries: #checks if is one of selected queries
                for doc in query_documents:
                    embeddings.append(doc['meta']['embedding_2d'])
                    title = doc["id"].replace(".pdf" , "")
                    title = title.replace("_", " ")
                    titles.append(title)
                    #titles.append(doc['meta']['title'])

        emb_df = pd.DataFrame(embeddings, columns = ['x','y'])
        emb_df['Title'] = titles   
        fig = px.scatter(emb_df, x='x', y='y', custom_data = ['Title'] )
        fig.update_traces(
        hovertemplate="<br> %{customdata}"
        )
        #fig.update_layout(transition=dict(duration=500, easing='cubic-in-out'))
        fig = self.fig_trace_update(fig)
        return fig

    #creates heatmap
    def update_heatmap_graph(self, selected_queries):
        texts = []
        if isinstance(selected_queries, dict):
            for points in selected_queries["points"]:
                title = points["customdata"][0]
                texts.append(self.title_content_map[title])
        else:
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
                        x=df.columns, y=["Doc {}".format(i) for i in range(len(texts))], color_continuous_scale='Agsunset', origin='upper')
        #fig.update_layout(title="Word Frequency Heatmap")
        fig.update_layout(                  
                    autosize = True,
                    margin=dict(l= 10, r = 10, t=5, b=5),
                    plot_bgcolor="rgba(242,242,242, 0.5)",
                    transition={
                            'duration': 1000,  # 1 second
                            'easing': 'cubic-in-out'
                    }
                    )
        return fig
    
    #makes wordcloud
    def update_wordcloud_graph(self, selected_queries):
        texts = []
        if isinstance(selected_queries, dict):
            for points in selected_queries["points"]:
                title = points["customdata"][0]
                texts.append(self.title_content_map[title])
        else:
            for query, query_documents in self.documents.items():
                if query in selected_queries: #checks if is one of selected queries
                    for doc in query_documents:
                        texts.append(doc['content'])
        
        texts = ' '.join(texts).split()

        stop_words = list(text.ENGLISH_STOP_WORDS.union(['additional', 'stopwords', 'if', 'needed']))        
        filtered_text = [word for word in texts if word.lower() not in stop_words and len(word) > 2]

        wordcloud = WordCloud(width=600, height=300, background_color='white').generate(" ".join(filtered_text)).to_image()

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
        if isinstance(selected_queries, dict):
            for points in selected_queries["points"]:
                title = points["customdata"][0]
                texts.append(self.title_content_map[title])
        else:
            for query, query_documents in self.documents.items():
                if query in selected_queries: #checks if is one of selected queries
                    for doc in query_documents:
                        texts.append(doc['content'])

        location_counts = Counter()
        for text in texts:
            location_counts.update(self.extract_locations(text))

        df = pd.DataFrame(location_counts.items(), columns=['country_code', 'count'])
        df['country'] = df['country_code'].apply(lambda x: pycountry.countries.get(alpha_3=x).name)

        fig = px.choropleth(df, locations='country_code', color='count',
                            hover_name='country', color_continuous_scale='Agsunset', projection='equirectangular',
                           )

        fig.update_geos(showcoastlines=True, coastlinecolor="Black",
                        showland=True, landcolor="lightgreen",
                        showocean=True, oceancolor="lightblue",
                        showlakes=False, lakecolor="lightblue",
                        showrivers=False, rivercolor="lightblue",
                        showframe = False, )

        fig.update_layout(margin=dict(l= 10, r = 10, t=5, b=5))
        return fig
    ##### Visualization Code End

    def update_graphs(self, selected_queries):
        trigger = ctx.triggered_id

        if len(selected_queries) !=0:

            if trigger != "umap":
                umap_fig = self.update_2d_graph(selected_queries)
                heatmap_fig = self.update_heatmap_graph(selected_queries)
                location_map_fig = self.update_location_map(selected_queries)
                wordcloud_src = self.update_wordcloud_graph(selected_queries)

                return umap_fig, heatmap_fig, location_map_fig, wordcloud_src
            
            else:
                heatmap_fig = self.update_heatmap_graph(selected_queries)
                location_map_fig = self.update_location_map(selected_queries)
                wordcloud_src = self.update_wordcloud_graph(selected_queries)

                return heatmap_fig, location_map_fig, wordcloud_src


        
        else:
            return {}, {}, {}, ''
            


def run_viz_server():
    dash_app = Dashboard()
    dash_app.run()
