import gradio as gr
import requests
import pandas as pd

class GradioSearchUI:
    def __init__(self, search_server_url):
        self.search_server_url =  search_server_url
        self.query_result_cache = {}
        self.previous_queries = []  # is a list of queries, not buttons
        self.interface = self.build_interface()
        
    def search_query(self, query):
        response = requests.post(f'{self.search_server_url}/search_query', json={"query": query})
        if response.status_code == 200:
            results = response.json()
            self.query_result_cache[query] = results
            return results
        else:
            return "Error retrieving results"
        
    def update_previous_queries(self):
        # Convert previous queries into Gradio buttons
        response = requests.get(f'{self.search_server_url}/get_previous_queries')
        if response.status_code == 200:
            self.previous_queries = response.json()
        else:
            print('WARNING: No previous queries found. If this is your first query please ignore this warning.')
            self.previous_queries = []

    def build_interface(self):
        with gr.Blocks() as interface:
            with gr.Tab("Search"):
                self.search_input = gr.Textbox(placeholder="Search...", label="")
                self.search_button = gr.Button("Search")
                self.search_output = gr.DataFrame(visible = False)
                
                def search(query):
                    results = self.search_query(query)
                    self.update_previous_queries()
                    if isinstance(results, dict):
                        documents = results["search"]["documents"]
                        data = []
                        for doc in documents:
                            data.append({
                                "id": doc["id"],
                                "Content": doc["content"],
                                "Score": doc["score"]
                            })
                        df = pd.DataFrame(data)
                        return gr.DataFrame(df, visible = True, wrap=False, column_widths = ['30%','60%','10%'])

                    else:
                        return gr.DataFrame(pd.DataFrame([], columns = ['No Results Returned.']), visible = True)
                    
                    #to do: implement previous queries, implement download, implement analysis tab refresh


                self.search_button.click(fn=search, inputs=self.search_input, outputs=self.search_output)
                    
            with gr.Tab("Analysis"):
                # This tab is left blank for now
                gr.Markdown("Analysis Tab Content")

        return interface

    def launch(self):
        self.interface.launch(share=False)

if __name__ == "__main__":
    search_server_url = 'http://127.0.0.1:5000'
    gradio_ui = GradioSearchUI(search_server_url)
    gradio_ui.launch()