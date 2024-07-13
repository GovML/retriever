import gradio as gr
import requests
import pandas as pd

class GradioSearchUI:
    def __init__(self, search_server_url, analysis_server_url):
        self.search_server_url =  search_server_url
        self.analysis_server_url = analysis_server_url
        self.current_results = pd.DataFrame([])
        self.previous_queries = []  # is a list of queries, not buttons
        self.interface = self.build_interface()
        
    def search_query(self, query):
        response = requests.post(f'{self.search_server_url}/search_query', json={"query": query})
        if response.status_code == 200: #should cover case when nothing is sent
            results = response.json()
            documents = results["search"]["documents"]
            qa_answer, score = results["qa"]['qa_answer'], results["qa"]['score']
            synthesized_qa = f"### Synthesized Answer: {qa_answer}" if score > 0.5 else f"### We could not find an answer contained within your documents: Confidence {round(score, 2)} - {qa_answer}"
            data = []
            for doc in documents:
                data.append({
                    "Document Name": doc["id"],
                    "Content": doc["content"],
                })
            self.current_results = pd.DataFrame(data)
            
            if [query] not in self.previous_queries:
                self.previous_queries.append([query])
            
            return gr.DataFrame(self.current_results, visible = True, wrap=False, column_widths = ['30%','60%','10%']), synthesized_qa, gr.DownloadButton(label=f"Download Results", visible=True), gr.Dataset(samples=self.previous_queries, components=[self.search_input], type="index", label="Queries")   
        else:
            return gr.DataFrame(pd.DataFrame([], columns = ['No Results Returned.']), visible = True), "### No Results Found", gr.DownloadButton(label=f"Download Results", visible=False), gr.Dataset(samples=self.previous_queries, components=[self.search_input], type="index", label="Queries")
    
    def download_results(self, query):
        sanitized_query = "".join(x for x in query if x.isalnum())
        gr.Info(f'Saving CSV {sanitized_query}.csv to current directory.')
        self.current_results.to_csv(f"./{sanitized_query}.csv")

    def update_previous_queries(self, id):
        # Convert previous queries into Gradio buttons
        input_query = self.previous_queries[id][0]
        search_output, qa_output, _, _ = self.search_query(input_query)
        return input_query, search_output, qa_output

    def build_interface(self):
        with gr.Blocks() as interface:
            with gr.Tab("SEARCH"):
                gr.Markdown("# Let's Get You Started With Search")
                self.search_input = gr.Textbox(placeholder="What would you like to know?", label="")
                self.search_button = gr.Button("Search")

                gr.Markdown("## Searched Queries")
                self.searched_queries = gr.Dataset(samples=self.previous_queries, components=[self.search_input], type="index", label="Queries")

                gr.Markdown("## Synthesized Outputs")
                self.qa_output = gr.Markdown("### Outputs will appear here.")
                self.search_output = gr.DataFrame(self.current_results, visible = True)
                self.download_button = gr.DownloadButton(label=f"Download Results", visible=False)
                

                self.searched_queries.click(self.update_previous_queries, inputs=[self.searched_queries], outputs=[self.search_input, self.search_output, self.qa_output])
                self.search_button.click(fn=self.search_query, inputs=self.search_input, outputs=[self.search_output, self.qa_output, self.download_button, self.searched_queries])
                self.download_button.click(fn = self.download_results, inputs = self.search_input)
                    
            with gr.Tab("ANALYSIS"):
                # This tab is left blank for now
                gr.Markdown("# Analysis Tab Content")
                src_val = self.analysis_server_url
                html = ("<iframe id=\"iframeid\" src=" + src_val + "\ width=1200 height=2000>")
                out1 = gr.HTML(html)

        return interface

    def launch(self):
        self.interface.launch(share=False)

def run_frontend():
    search_server_url = 'http://127.0.0.1:5000'
    analysis_server_url = 'http://127.0.0.1:8055'
    gradio_ui = GradioSearchUI(search_server_url, analysis_server_url)
    gradio_ui.launch()