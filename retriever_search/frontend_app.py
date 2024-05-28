import argparse
import sys
import os
import gradio as gr
import warnings
import pandas as pd
import json
import requests
import os
import time

warnings.filterwarnings("ignore")
keyword = []


class gradio_app:
    def __init__(self, search_link, analysis_link):
        self.search_link = search_link
        self.analysis_link = analysis_link
        self.qa_answer_value = "### You havent asked any questions yet"
        self.input_query = ''

    def load_example(self, example_id):
        global keyword

        data_values = { "query": str(keyword[example_id][0])}
        responses = requests.post(self.search_link + "/search_query", json=data_values)
        qa_answer = requests.get(self.search_link + "/get_qa_answer")
        self.qa_answer_value = "### " + qa_answer.json()[0]['qa_answer']
        df = pd.read_json(responses.json(), orient='records')
        print(keyword[example_id])
        return keyword[example_id][0], df, gr.Markdown(self.qa_answer_value)

    def search_query(self, input_query):
        global keyword
        data_values = { "query": str(input_query)}
        responses = requests.post(self.search_link + "/search_query", json=data_values)
        qa_answer = requests.get(self.search_link + "/get_qa_answer")
        print(qa_answer)
        self.qa_answer_value = "### " + qa_answer.json()[0]['qa_answer']
        df = pd.read_json(responses.json(), orient='records')

        #df, doc_ids = self.search_code_arxiv(responses)
        if [input_query] not in keyword:
            keyword.append([input_query])

        #file = open('./plotly_dash_viz/main.py', 'r')
        #file.close()
        if not os.path.isdir('./output/'):
            os.makedirs('./output/')
        df.to_csv('./output/' + input_query.strip('?') + '.csv')
        filepath = './output/' + input_query.strip('?') + '.csv'
        name = input_query.strip('?') + '.csv'
        self.input_query = input_query
        return df, keyword, gr.Markdown(self.qa_answer_value), gr.DownloadButton(label=f"Download {name}", value=filepath, visible=True)

    def export_data(self, df):
        query = self.input_query.strip('?')
        filepath = './output/' + query + '.csv'
        os.remove(filepath)
        return gr.DownloadButton(label=f"Download the file", visible=False)


    def gradio_launch(self):
        with gr.Blocks() as demo:
            with gr.Tabs() as tabs:
                with gr.TabItem("SEARCH", id = 0):
                    gr.Markdown("# Let's Get You Started With Search")
                    gr.Markdown("# Search Query")
                    inp = gr.Textbox(label="Enter your query here - ")
                    with gr.Row():
                        btn = gr.Button("Search")
                    
                    gr.Markdown("## Searched Queries")
                    examples = gr.Dataset(samples=keyword, components=[inp], type="index", label="Queries")

                    gr.Markdown("## Synthesized Results")
                    qa_answer_markdown = gr.Markdown(self.qa_answer_value)

                    gr.Markdown("## Other Relevant Papers")
                    out = gr.DataFrame(pd.DataFrame([], columns = ['Title']), wrap = True)
                    with gr.Row():
                        btn1 = gr.DownloadButton("Download the file", visible=False)
                        #btn1 = gr.Button("Export Results")
                    examples.click(self.load_example, inputs=[examples], outputs=[inp, out, qa_answer_markdown])
                    btn.click(fn=self.search_query, inputs=inp, outputs=[out, examples, qa_answer_markdown, btn1])
                    btn1.click(fn=self.export_data, inputs=[out], outputs=[btn1])
                    #btn1.click(fn=self.export_data, inputs=[out, inp], outputs=[])
                    #os.remove(filepath)
                with gr.TabItem("ANALYSIS", id = 1):
                    gr.Markdown("# Analyse your searched data")
                    
                    src_val = self.analysis_link
                    html = ("<iframe id=\"iframeid\" src=" + src_val + "\ width=1200 height=2000>")
                    out1 = gr.HTML(html)


        demo.launch(share=False) 

def run_frontend():
    gradio_obj = gradio_app("http://127.0.0.1:5000", "http://127.0.0.1:8055")
    gradio_obj.gradio_launch()
