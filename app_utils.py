# Load EDA pkgs
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.use("Agg")

# load NLP pkgs
import spacy
!python3 -m spacy download en_core_web_sm

nlp = spacy.load("en_core_web_sm")
from spacy import displacy
from textblob import TextBlob

# load text cleaning pkgs
import neattext as nt
import neattext.functions as nfx

# utils
from collections import Counter
import base64
from datetime import datetime

now = datetime.now()
timestr = now.strftime("%m/%d/%Y-%H:%M:%S")

# file processing pkgs
import docx2txt
import pdfplumber

from PyPDF2 import PdfFileReader

# define dunction to read pdf file
def read_pdf(file):
    pdfReader = PdfFileReader(file)
    count = pdfReader.numPages
    all_page_text = ""
    for i in range(count):
        page = pdfReader.getPage(i)
        all_page_text += page.extractText()

    return all_page_text


def text_analyzer(my_text):
    docx = nlp(my_text)
    allData = [
        (
            token.text,
            token.shape_,
            token.pos_,
            token.tag_,
            token.lemma_,
            token.is_alpha,
            token.is_stop,
        )
        for token in docx
    ]
    df = pd.DataFrame(
        allData, columns=["Token", "Shape", "PoS", "Tag", "Lemma", "IsAlpha", "IsStop"]
    )
    return df


def get_entities(my_text):
    docx = nlp(my_text)
    entities = [(entity.text, entity.label_) for entity in docx.ents]
    return entities


HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>"""

# define function to gey entities with rendered format
def render_entities(raw_text):
    docx = nlp(raw_text)
    html = displacy.render(docx, style="ent")
    html = html.replace("\n\n", "\n")
    result = HTML_WRAPPER.format(html)
    return result


# define function to get most common tokens
def get_most_common_tokens(my_text, num=4):
    word_tokens = Counter(my_text.split())
    most_common_tokens = dict(word_tokens.most_common(num))
    return most_common_tokens


# define function to get sentiment
def get_sentiment(my_text):
    blob = TextBlob(my_text)
    sentiment = blob.sentiment
    return sentiment
