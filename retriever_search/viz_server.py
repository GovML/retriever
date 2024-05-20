#Importing the Libraries
import dash
from flask import Flask
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
#Test
import requests
from retriever_search.callbacks import get_callbacks
from retriever_search.layout import Layout
import json
import warnings
import ast  
import nltk

warnings.filterwarnings("ignore")

server = Flask(__name__)
app = dash.Dash(__name__, server=server)

def get_init_df():
      return pd.DataFrame(columns = ["title", "title_abs", "doc_ids", "keyword", "emb1", "emb2"])

def init_topicModel():
   this_dir, this_filename = os.path.split(__file__)
   with open(this_dir + '/TopicModellingInit.html', 'r') as f:
      vis_html = f.read()
   #vis_html = importlib.resources.read_binary("retriever_search", "TopicModellingInit.html")
   return vis_html

vis_html = init_topicModel()
df = get_init_df()
keyword_data = []



def get_json(path):
   with open(path, "r") as f:
      return json.load(f) 


def run_viz_server():
    nltk.download('stopwords')
    call_b = get_callbacks(app, df, keyword_data)
    app.layout = Layout(df, vis_html, keyword_data).layout
    app.run_server(debug=False, host = "127.0.0.1",port=8055)