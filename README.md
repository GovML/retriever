# Retriever


## Install 

Recommended to use a virtual environment using venv - 

```bash
$ python -m venv new_env
$ source new_env/bin/activate
$ pip install -e .
```

## Search server and frontend run

```bash
>>> from retriever_search import search_server
>>> from retriever_search import frontend_app as fp
>>> fp.run_frontend()
>>> search_server.run_search_server('path_to_folder', 'save_json_path')
```