from setuptools import setup

from search_src import __version__

setup(
    name='search_src',
    version=__version__,

    url='https://github.com/GovML/retriever.git',
    author='Sidharth Kathpal',
    author_email='kathpal.sid@gmail.com',

    py_modules=['search_src'],
)