from langchain_community.document_loaders import PyMuPDFLoader
import tempfile
import requests

def load_docs(url):
    response = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    loader = PyMuPDFLoader(tmp_path)
    return loader.load()
