from setuptools import setup, find_packages

from retriever_search import __version__

setup(
    name='retriever_search',
    version=__version__,

    url='https://github.com/GovML/retriever.git',
    author='Sidharth Kathpal',
    author_email='kathpal.sid@gmail.com',
    description='Local retriever search for your use',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    py_modules=['retriever_search'],
    install_requires=[
        'haystack-ai==2.0.0',
        'pymupdf==1.22.5',
        'sentence-transformers==2.5.1',
        'umap==0.1.1',
        'umap-learn==0.5.5',
        'qdrant-haystack==3.0.0',
        'Flask==3.0.2',
        'pandas==2.2.1',
        'torch==2.2.1',
        'accelerate==0.27.2',
        'gradio==4.21.0'
    ],
    packages=find_packages(),
)