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


# set up server
server = Flask(__name__)
app = dash.Dash(__name__, server=server, external_stylesheets=['https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css'])

#insert post
search_results = requests.get("http://127.0.0.1:5000/search_results").json()
queries = {s['query']:i for i, s in enumerate(search_results)}

#util functions to get data
def get_text(query_index):
    return [doc['Content'] for doc in ast.literal_eval(search_results[query_index]['data'])]

def get_doc_ids(query_index):
    return [doc['Title'] for doc in ast.literal_eval(search_results[query_index]['data'])]
            
def get_embeddings(query_index):
    embeddings = requests.get("http://127.0.0.1:5000/search_return_embeddings").json()
    #list of 2d embeddings
    embs = [e['embeddings'] for e in embeddings[query_index]['embedding_2d']]
    return embs

#plotting helper functions
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(text))
    return wordcloud

def wordcloud_image(wordcloud):
    img = io.BytesIO()
    plt.figure(figsize=(10, 5), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(img, format='PNG')
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

def generate_word_frequency_heatmap(texts):
    [print(t[0:10]) for t in texts]
    stop_words = list(text.ENGLISH_STOP_WORDS.union(['additional', 'stopwords', 'if', 'needed']))
    vectorizer = text.CountVectorizer(max_features=50, stop_words=stop_words)
    X = vectorizer.fit_transform(texts)
    df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
    fig = px.imshow(df, labels=dict(x="Words", y="Documents", color="Frequency"),
                    x=df.columns, y=["Doc {}".format(i) for i in range(len(texts))])
    fig.update_layout(title="Word Frequency Heatmap")
    return fig

#app contents
app.layout = app.layout = html.Div([
    dcc.Dropdown(list(queries.keys()), list(queries.keys())[0], id='query-selector'),
    dcc.Graph(id='graph-content'),
    html.Img(id='wordcloud'),
    dcc.Graph(id='heatmap')
])

def update_2d_graph(value):
    embeddings = get_embeddings(queries[value])
    
    emb_df = pd.DataFrame(embeddings, columns = ['x','y'])
    emb_df['Title'] = get_doc_ids(queries[value])    
    fig = px.scatter(emb_df, x='x', y='y', custom_data=['Title'])
    fig.update_traces(
    hovertemplate="<br>".join([
        "%{customdata[0]}",
    ])
    )
    fig.update_layout(transition=dict(duration=500, easing='cubic-in-out'))

    return fig

def update_wordcloud_graph(value):
    texts = get_text(queries[value])
    wordcloud = generate_wordcloud(texts)
    img_src = "data:image/png;base64,{}".format(wordcloud_image(wordcloud))
    return img_src

def update_heatmap_graph(value):
    texts = get_text(queries[value])
    heatmap_fig = generate_word_frequency_heatmap(texts)
    return heatmap_fig
@callback(
    Output('graph-content', 'figure'),
    Output('wordcloud', 'src'),
    Output('heatmap', 'figure'),
    Input('query-selector', 'value')
)
def update_graph(value):
    emb_2d_graph_fig = update_2d_graph(value)
    img_src = update_wordcloud_graph(value)
    heatmap_fig = update_heatmap_graph(value)
    return emb_2d_graph_fig, img_src, heatmap_fig

app.run_server(debug=False, host = "127.0.0.1",port=8055)
