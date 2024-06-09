import dash
from dash import dcc, html
import json
import requests

class Layout():
    def __init__(self, df , vis_html, keyword_data):
        self.df = df
        self.vis_html = vis_html
        self.keyword_data = keyword_data
        self.layout = html.Div(
        [
            html.Div([
           
            html.Br(),
            html.H2("WordCloud of the Seletected Points",style = {'color': "#1F618D", 'font-family': 'Roboto, sans-serif','margin-right': '0%'}),
            html.Img(id ="word-cloud", disable_n_clicks = True, alt = "WordCloud", style = {'margin-top':'0px','margin-left':'0px','height': '400px', 'width':'400px'})], style={'float': 'right','margin-right': '5%', 'margin-top': '0px'}),

            dcc.Graph(id="live-graph", style= {'margin-left' : '0px',"width": "50%"} ),
            dcc.Location(id= "url", refresh = False),
            html.A(html.Button('Refresh Data', id = "refresh", n_clicks = 0,style={
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
                            })),
            html.H2("Select Keyword: ", style = {'color': "#1F618D", 'font-family': 'Roboto, sans-serif'}),
            dcc.Checklist(id = "keyword",options = [i["query"] for i in self.keyword_data],style={'color': '#000000',  'padding': '10px','color': "#1F618D", 'font-family': 'Roboto, sans-serif', 'font-size': '20px'}, inline=True ),

            
            html.Div([
            html.H1("Topic Modelling",style = {'color': "#1F618D", 'font-family': 'Roboto, sans-serif'})], style={'float': 'left', 'margin-top': '0px', 'width': '100%'}),
            
            html.Br(),
            
            html.Div([
                html.Iframe(
                    id = "topic_model",
                    srcDoc=vis_html,
                    width='100%',
                    height=1000,
                )
            ]),
                
        ]
        )

    def get_json(self, path):
      with open(path, "r") as f:
        return json.load(f) 
      
    def update_layout_df(self,df):
       self.df = df
    
    