import pandas as pd
from transformers.pipelines import pipeline
import streamlit as st

st.cache(show_spinner=False)
def load_model():
    hg_comp = pipeline('question-answering', model="bert-large-cased-whole-word-masking-finetuned-squad", tokenizer="bert-large-cased-whole-word-masking-finetuned-squad")
    return hg_comp

data = pd.read_csv('examples.csv')

nlp_pipe = load_model()
st.header("QA App")

question = st.text_input(label='Question')
text = st.text_area(label="Reference")


if (not len(text)==0) and (not len(question)==0):
    x_dict = nlp_pipe(context=text,question=question)
    st.text('Answer :'+ x_dict['answer'])
