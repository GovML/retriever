<div align="center">

# Retriever

### To setup your own local search you can now use this repo.

<p>
<img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/GovML/retriever" />
<img alt="GitHub Last Commit" src="https://img.shields.io/github/last-commit/GovML/retriever" />
<img alt="GitHub Repo Size" src="https://img.shields.io/github/repo-size/GovML/retriever" />
<img alt="GitHub Issues" src="https://img.shields.io/github/issues/GovML/retriever" />
<img alt="GitHub Pull Requests" src="https://img.shields.io/github/issues-pr/GovML/retriever" />
<img alt="Github License" src="https://img.shields.io/badge/License-Apache-yellow.svg" />
</p>

</div>
<iframe width="640" height="300" src="https://www.loom.com/embed/da7c3ac5d5934689bdb18fd2d8bf9643?sid=cb6fe700-7815-4c5c-b3fa-9dae0ce73bf4" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen></iframe>
-----
## Install 

Recommended to use a virtual environment using venv - 

```bash
$ python -m venv new_env
$ source new_env/bin/activate
```
For installing from the github repo - 
```bash
$ pip install -e .
```
For installing from the pip - 
```bash
$ pip install retriever-search
```

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

