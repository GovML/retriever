from setuptools import setup

from search import __version__

setup(
    name='search',
    version=__version__,

    url='https://github.com/GovML/retriever.git',
    author='Sidharth Kathpal',
    author_email='kathpal.sid@gmail.com',

    py_modules=['search'],
)