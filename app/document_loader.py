from langchain_community.document_loaders import PyMuPDFLoader
import tempfile, requests

def load_pdf_from_url(url):
    r = requests.get(url)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(r.content)
        path = tmp.name
    return PyMuPDFLoader(path).load()
