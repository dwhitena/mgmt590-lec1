import pandas as pd
import transformers
from transformers.pipelines import pipeline
from transformers import DebertaTokenizer, DebertaModel


hg_comp = pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='bert-base-cased')

data = pd.read_csv('examples.csv')

for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = hg_comp({'question': question, 'context': context})['answer']
    print(answer)
