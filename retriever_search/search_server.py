import sys
import os
import pandas as pd
from flask import Flask, jsonify, request
import torch
from retriever_search.search_app import search_app


app = Flask(__name__)

search_data = []
doc_ids = []
embedding_2d = []
qa_answers = []
query_list = []

@app.route('/search_results', methods=['GET'])
def get_results():
    return jsonify(search_data)

@app.route('/search_docids', methods=['GET'])
def get_docids():
    return jsonify(doc_ids)

@app.route('/search_return_embeddings', methods=['GET'])
def get_embeddings():
    return jsonify(embedding_2d)

@app.route('/get_qa_answer', methods=['GET'])
def get_qa_answer():
    return jsonify(qa_answers)

@app.route('/get_queries', methods=['GET'])
def get_queries():
    return jsonify(query_list)

@app.route('/search_query', methods=['POST'])
def add_query():
    global qa_answers
    global main_pipe
    qa_answers = []
    query = request.get_json()['query']
    #df, doc_id_values = search_code(main_pipe, query)
    df, doc_id_values, emb2d, qa_answer = new_search_code(main_pipe, query)
    data_values = df.to_json(orient="records")
    search_data.append({"query": query, "data": data_values})
    doc_ids.append({"query": query, "doc_id": doc_id_values})
    embedding_2d.append({"query": query, "embedding_2d": emb2d})
    query_list.append({"query": query})
    qa_answers.append({'qa_answer': qa_answer})
    print('qa_answer', qa_answer)
    return jsonify(data_values)

def new_search_code(main_pipe, input_query):
    torch.cuda.empty_cache()
    responses = main_pipe.run(query=str(input_query))
    print("HELLO THERE WE REACHED HERE")
    documents = responses['search']['documents']
    if responses['qa']['answers'][0].data is not None:
        if responses['qa']['answers'][0].score >= 0.50:
            answer = responses['qa']['answers'][0].data
            source = responses['qa']['answers'][0].document.id
            qa_answer = f'{answer}, According to {source}.'
        else:
            qa_answer = "Sorry I could not find information for this. I only have data on your pc. Which may not contain an answer for your question."
    else:
        qa_answer = "Sorry I could not find information for this. I only have data on your pc. Which may not contain an answer for your question."
    doc_ids = []
    titles = []
    abstracts = []
    arxiv_links = []
    embedding_2d = []
    print('new search code: ', qa_answer)

    for i in range(min(len(documents), len(documents))):
        doc = documents[i]
        doc_id = doc.id
        #print(doc.meta['title'])
        if doc.meta['title'] is not None:
            title = doc.meta['title']
            arxiv_link = f'https://arxiv.org/abs/{doc_id}'
        else:
            title = doc.id.rstrip('.pdf')
            arxiv_link = doc_id
        abstract = doc.content
        emb_2d = doc.meta['embedding_2d']
        titles.append(title)
        abstracts.append(abstract)
        arxiv_links.append(arxiv_link)
        embedding_2d.append({"doc_ids" : doc_id, "embeddings" : emb_2d})
        doc_ids.append(doc_id)

    df = pd.DataFrame(list(zip(titles, abstracts, arxiv_links)), columns = ['Title','Content','Link'])
    return df, doc_ids, embedding_2d, qa_answer


def run_search_server(inputDirectory, json_save_path, device):
    global main_pipe
    print(device)
    search_obj = search_app(inputDirectory, json_save_path, device = device)
    main_pipe = search_obj.new_search_run()
    app.run(debug=False, host='127.0.0.1', threaded=True)