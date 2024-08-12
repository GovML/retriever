<div align="center">

# Retriever

### Visually search and analyze your documents, entirely locally.

<p>
<img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/GovML/retriever" />
<img alt="GitHub Last Commit" src="https://img.shields.io/github/last-commit/GovML/retriever" />
<img alt="GitHub Repo Size" src="https://img.shields.io/github/repo-size/GovML/retriever" />
<img alt="GitHub Issues" src="https://img.shields.io/github/issues/GovML/retriever" />
<img alt="GitHub Pull Requests" src="https://img.shields.io/github/issues-pr/GovML/retriever" />
<img alt="Github License" src="https://img.shields.io/badge/License-Apache-yellow.svg" />
</p>
<img src="./retriever.gif"/>
</div>

## Install 
Options:
1) Install with pip (Stable Release)
```bash
$ pip install retriever-search
```
2) Install from Github Repo (Latest Release)
```bash
$ git clone https://github.com/GovML/retriever.git
$ pip install -e .
```
We recommended using a virtual environment for all dependency installations. Before installing our repo, you can use venv to isolate the various packages installed in this environment to prevent conflicts with versions already installed on your computer.

```bash
$ python -m venv new_env
$ source new_env/bin/activate
```

# Quickstart - Launching Retriever
fsdfsdf

# In-Depth Usage Guide
## Search server

```bash
>>> from retriever_search import search_server
>>> search_server.run_search_server('input_directory', 'input_json', 'json_save_path', 'embedding_model', 'qa_model', device='cpu')
```

## Search parameter meanings

- input_directory -- The directory holding your files
- input_json -- pre saved json file from earlier runs can be used for faster loading
- json_save_path -- pass for saving the embeddings to a json can be used later as input_json
- embedding_model -- pick the embedding model you want to we use Spectre model as a default
- qa_model -- you can currently pick between tiny, medium and large

## Frontend and Vizualisation run

```bash
>>> from retriever_search import frontend_app as fp
>>> fp.run_frontend()
```


## Where to access the frontend 

Access via the following URL - http://127.0.0.1:7860 
This URL would work for your local setup only


## Tickets

1.0.0
- [ ] Quickstart Dataset
- [ ] Make LDA visualization update
- [ ] QA Model Improvements

