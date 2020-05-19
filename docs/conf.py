import datetime
import os

project = 'PyBlaze'
version = os.getenv('CIRCLE_TAG', '0.0.0')
release = os.getenv('CIRCLE_TAG', '0.0.0')

author = 'Oliver Borchert'
copyright = f'{datetime.datetime.now().year}, {author}' # pylint: disable=redefined-builtin

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.mathjax',
    'sphinx_rtd_theme',
    'nbsphinx',
]
exclude_patterns = []

autodoc_member_order = 'bysource'
autodoc_default_options = {
    'special-members': '__init__',
}

html_theme = 'sphinx_rtd_theme'
html_static_path = []

nbsphinx_execute = 'never'
nbsphinx_prompt_width = '0'
