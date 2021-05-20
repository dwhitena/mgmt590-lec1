import pandas as pd
from transformers.pipelines import pipeline
pip install gradio
import gradio as gr


def answer_question(question: str, paragraph: str):
    input = {
        "question": question,
        "context": paragraph
    }
    hg_comp = pipeline('question-answering', model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad") #Using the Distilbert-base-uncased-distilled-squad model
    return hg_comp(input)['answer']


gr.Interface(fn=answer_question, inputs=["textbox", "text"], outputs= "text").launch(share = True, inbrowser = True)
