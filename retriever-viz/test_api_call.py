import requests
import pandas as pd
import ast

import plotly.express as px
import dash
from dash import Dash, html, dcc, callback, Output, Input

from flask import Flask

# set up server
server = Flask(__name__)
app = dash.Dash(__name__, server=server)


search_results = requests.get("http://127.0.0.1:5000/search_results").json()
queries = {s['query']:i for i, s in enumerate(search_results)}

def get_doc_ids(query_index):
    return [doc['Title'] for doc in ast.literal_eval(search_results[query_index]['data'])]
            
def get_embeddings(query_index):
    embeddings = requests.get("http://127.0.0.1:5000/search_return_embeddings").json()
    #list of 2d embeddings
    embs = [e['embeddings'] for e in embeddings[query_index]['embedding_2d']]
    return embs

#app contents
app.layout = app.layout = html.Div([
    dcc.Dropdown(list(queries.keys()), list(queries.keys())[0], id='query-selector'),
    dcc.Graph(id='graph-content')
])


@callback(
    Output('graph-content', 'figure'),
    Input('query-selector', 'value')
)
def update_graph(value):
    embeddings = get_embeddings(queries[value])
    
    emb_df = pd.DataFrame(embeddings, columns = ['x','y'])
    emb_df['Title'] = get_doc_ids(queries[value])    
    fig = px.scatter(emb_df, x='x', y='y', custom_data=['Title'])
    fig.update_traces(
    hovertemplate="<br>".join([
        "%{customdata[0]}",
    ])
    )
    return fig

#add a wordcloud

#

app.run_server(debug=False, host = "127.0.0.1",port=8055)
