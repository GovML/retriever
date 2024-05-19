#Importing the Libraries
import dash
from dash.dependencies import Output, Input,State
import plotly.graph_objs as go
from wordcloud import WordCloud, STOPWORDS
import pyLDAvis.lda_model
import json
import ast
import textwrap
import re
from nltk.corpus import stopwords
import pickle
import matplotlib
import pandas as pd
import requests
from tqdm import tqdm
import os


class get_callbacks():
    def __init__(self, app, df, keyword_data):
        self.init_df = df
        self.df = df
        self.title_abs = {}
        self.keyword_data = keyword_data
        self.keyword = list(set([i["query"].capitalize() for i in self.keyword_data]))
        self.app = app
        self.ldaModel, self.dtm_tf, self.tf_vectorizer = self.init_data()
        self.get_callback()
        self.n_clicks = 0
        self.search_results = None
        #self.document_store = self.get_retriever(outputDirectory= "./500K_test/" )

    
    def init_data(self):
        
        this_dir, this_filename = os.path.split(__file__)
        print(this_dir)
        #Loading the lda_model for topic_modeling
        with open(this_dir + "/data/LDAModel.pkl", "rb") as f:
            ldaModel = pickle.load(f)

        #Loading the Count vectorizer array
        with open(this_dir + "/data/EmbeddedVectors.pkl", "rb") as f:
            dtm_tf = pickle.load(f)

        #Loading the Count Vectorizer object
        with open(this_dir + "/data/CountVectorizer.pkl", "rb") as f:
            tf_vectorizer = pickle.load(f)

        #ldaModel = pickle.load(importlib.resources.read_text("retriever_search", "LDAModel.pkl"))
        #dtm_tf = pickle.load(importlib.resources.read_text("retriever_search", "mbeddedVectors.pkl"))
        #tf_vectorizer = pickle.load(importlib.resources.read_text("retriever_search", "CountVectorizer.pkl"))

        return ldaModel, dtm_tf, tf_vectorizer



    def get_json(self,path):
        with open(path, "r") as f:
            return json.load(f) 
        
    def get_title_abs_data(self):
        
        if not len(self.search_results):
            pass

        for data in self.search_results:
            self.title_abs[data["query"]] = []
            content_data = ast.literal_eval(data["data"])
            for i in range(len(content_data)):
                data_ = content_data[i]
                data_["Title"] = re.sub(r'\n', ' ', data_["Title"])
                data_["Content"] = re.sub(r'\n', ' ', data_["Content"])
                self.title_abs[data["query"]].append(data_)




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
                transition={
                        'duration': 1000,  # 1 second
                        'easing': 'cubic-in-out'
                }
                )
        
        if paper_len == 0:
            fig.update_layout(template = "plotly_white")


        return fig
    def update_callback_df(self, df):
        self.df = df
        print(f"updating the df to {df.shape}")

    
    def get_dataframe(self):
        ''' Takes the json of kayword and doc ids and create dataframe compatible for the viz'''
        df = self.df
        plot_data = {"title": [], "title_abs": [], "doc_ids": [], "keyword": [], "emb1": [], "emb2": []}


        for idx in range(len(self.keyword_data)):
            for emb, data in zip(self.keyword_data[idx]["embedding_2d"], eval(self.search_results[idx]["data"])):
                 
                 plot_data["title_abs"].append(data["Content"].replace("\n", ""))
                 plot_data["title"].append(data["Title"])
                 plot_data["keyword"].append(self.keyword_data[idx]["query"].lower())
                 plot_data["emb1"].append(emb["embeddings"][0])
                 plot_data["emb2"].append(emb["embeddings"][1])
                 plot_data["doc_ids"].append(emb["doc_ids"])

        return pd.DataFrame(plot_data)

    def plot_scatter(self,df):
        '''Function to plot the scatter plot of all the papers that takes
        Input: DataFrame
        Output: Scatter-plot Figure'''

        fig_main = go.Figure()
        # fig_main.add_trace(go.Scatter(x = df['emb1'], y = df['emb2'],  mode = 'markers', marker_color = df['color_code'], opacity = 1, text = df['title'], hovertemplate = 'Title: %{text} <br>' )) # + 'Author: %{customdata}<extra></extra>' # customdata = df['authors']
        # fig_main.add_trace(go.Scatter(visible = False, x = df['emb1'], y = df['emb2'],  mode = 'markers', marker_color = df['color_code'], opacity = 0.5, text = df['title'], hovertemplate = 'Title: %{text} <br>')) # + Author: %{customdata}<extra></extra> # customdata= df['authors']
        fig_main.add_trace(go.Scatter(x = df['emb1'], y = df['emb2'],  mode = 'markers', marker_color = "#1F618D", opacity = 1, text = df['title_abs'], hovertemplate = 'Title: %{customdata} <br>', customdata = df["title"] )) # + 'Author: %{customdata}<extra></extra>' # customdata = df['authors']
        fig_main.add_trace(go.Scatter(visible = False, x = df['emb1'], y = df['emb2'],  mode = 'markers', marker_color = "#1F618D", opacity = 0.5, text = df['title_abs'], hovertemplate = 'Title: %{customdata} <br>',customdata = df["title"])) # + Author: %{customdata}<extra></extra> # customdata= df['authors']
        fig_main.data[-1].visible = True
        fig_main.data[-2].visible = True
        fig_main = self.fig_trace_update(fig_main)
        return fig_main

    ## Callback for animation
    def get_callback(self):

        #Callback to update the wordcloud
        @self.app.callback(Output('word-cloud', 'src'), 
                    [Input('live-graph','selectedData')],
                    [Input("refresh", "n_clicks")],
                    # [Input("search-btn", "n_clicks")]
                    )
        def get_wordcloud(selected_data, n_clicks):
            '''Callback function to generate the wordcloud
            Input: selected datapoints from the plotly graph
            Output: wordCloud Image
            '''
            full_text = ""
            stopword_set = set(stopwords.words('english') + list(STOPWORDS) + ['[SEP]', 'using','based','well','used','may', 'SEP', 'sep'])

            #When the user selects some points in the plot
            if selected_data is not None and len(selected_data["points"]):
                for pt in selected_data["points"]:
                    title = pt["text"]  
                    # text_data = self.df.loc[self.df["title"] == title, "title_abs"].values
                    full_text += f"{title} "
                cloud_no_stopword = WordCloud(background_color='white', stopwords=stopword_set, colormap=matplotlib.colormaps['ocean'],
                                                            width=200, height=200, repeat=True).generate(full_text)
            
            if full_text == "":
                if len(self.df["title_abs"]):
                    full_text = " ".join(self.df["title_abs"].values)
                    cloud_no_stopword = WordCloud(background_color='white', stopwords=stopword_set, colormap=matplotlib.colormaps['ocean'],
                                                            width=200, height=200, repeat=True).generate(full_text)
                else:
                    full_text = "No_Data"
                    cloud_no_stopword = WordCloud(background_color='white', stopwords=stopword_set, colormap=matplotlib.colormaps['ocean'],
                                                            width=200, height=200, repeat=False,min_font_size=20, max_font_size=20, prefer_horizontal=1.0).generate(full_text)
            
                
           
            image = cloud_no_stopword.to_image()
            return image

        #callback for topic modelling model
        @self.app.callback(Output("topic_model","srcDoc"), 
                    [Input('live-graph','selectedData')],
                    )
        def get_topic_viz(selected_data):

            abs_data = []
            if selected_data is not None and len(selected_data["points"]):
                for pt in selected_data["points"]:
                    abs_data.append(pt["text"])


            data = self.tf_vectorizer.transform(abs_data)
            if data.shape[0] == 0:
                data = self.dtm_tf

            vis_lda_TF =  pyLDAvis.lda_model.prepare(self.ldaModel, data, self.tf_vectorizer)

            vis_html = pyLDAvis.prepared_data_to_html(vis_lda_TF)

            return vis_html

        
        @self.app.callback(Output("live-graph", "figure"),
                           Input("keyword","value"))
        def update_graph_with_new_data(value):  
            if value is None or len(value) == 0:
                self.df = self.init_df
                return self.plot_scatter(self.init_df)

            value = [i.lower() for i in value]

            key_data = [i for i in self.keyword_data if i["query"] in value]
            df = self.get_dataframe()
            self.df = df
            
            df = self.df.loc[df["keyword"].isin(value)]
            # print(df)
            fig = self.plot_scatter(df)
            return fig
        
        
        @self.app.callback(Output("keyword", "options"),
                           Input("refresh", 'n_clicks'))
        def update_json_keyword(n_clicks):
            self.keyword_data = requests.get("http://127.0.0.1:5000/search_return_embeddings").json()
            self.search_results = requests.get("http://127.0.0.1:5000/search_results").json()
            self.keyword = list(set([i["query"].capitalize() for i in self.keyword_data]))
            self.get_title_abs_data()
            self.df = self.get_dataframe()
            return self.keyword