import numpy
from click import Tuple
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from dataclasses import dataclass
import tempfile
import uvicorn

app = FastAPI()
templates = Jinja2Templates(directory="templates")


@dataclass
class TFIDFToken:
    word: str
    tf: int
    idf: float


@dataclass
class AnalyzedDocument:
    name: str
    tokens: List[TFIDFToken]


@app.get("/", response_class=HTMLResponse)
async def homepage(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})


async def parseDocuments(documents: List[UploadFile] = File(...)):
    temp_paths = []
    for doc in documents:
        tmp = tempfile.NamedTemporaryFile(delete=False)
        content = await doc.read()
        tmp.write(content)
        tmp.flush()
        temp_paths.append((tmp.name, doc.filename))
    return temp_paths


def fill_result(idf_vals: numpy.ndarray, temp_paths: List[Tuple], tf_matrix: numpy.ndarray, words: numpy.ndarray):
    analyzed_docs = []
    for i, (_, doc_name) in enumerate(temp_paths):
        entries = [
            TFIDFToken(word, int(tf), round(idf, 3))
            for word, tf, idf in zip(words, tf_matrix[i], idf_vals)
            if tf != 0
        ]
        sorted_tokens = sorted(entries, key=lambda x: x.idf, reverse=True)[:50]
        analyzed_docs.append(AnalyzedDocument(doc_name, sorted_tokens))
    return analyzed_docs


@app.post("/process", response_class=HTMLResponse)
async def process_documents(request: Request, documents: List[UploadFile] = File(...)):
    vectorizer = TfidfVectorizer(input="filename", norm=None)
    temp_paths = await parseDocuments(documents)
    tfidf_matrix = vectorizer.fit_transform([path for path, _ in temp_paths])
    words = vectorizer.get_feature_names_out()
    idf_vals = vectorizer.idf_
    tf_matrix = tfidf_matrix.toarray() / idf_vals
    result = fill_result(idf_vals, temp_paths, tf_matrix, words)
    return templates.TemplateResponse("results.html", {
        "request": request,
        "documents": result
    })


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
