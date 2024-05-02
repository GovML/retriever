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

## Search server and frontend run

```bash
>>> from retriever_search import search_server
>>> from retriever_search import frontend_app as fp
>>> fp.run_frontend()
>>> search_server.run_search_server('path_to_folder', 'save_json_path', device='cpu')
```

## Vizualisation run

Currently run from the git repo

```bash
$ cd retriever
>>> from retriever_search import viz_server as vs
>>> vs.run_viz_server()
```

## Where to access the frontend 

Access via the following URL - http://127.0.0.1:7860 
This URL would work for your local setup only
