import pandas as pd
from transformers.pipelines import pipeline

hg_comp =  pipeline('question-answering', model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")
hg_comp_fine_tuned = pipeline('question-answering', model="mrm8488/bert-multi-cased-finetuned-xquadv1", tokenizer="distilbert-base-uncased-distilled-squad")
german_translator = pipeline("translation_en_to_de")
french_translator = pipeline("translation_en_to_fr")

data = pd.read_csv('examples.csv')

for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer = hg_comp({'question': question, 'context': context})['answer']
    print(answer)
    answer_dict = hg_comp({'question': question, 'context': context})
    answer_finetuned = hg_comp_fine_tuned({'question': question, 'context': context})

answer =''
if answer_dict['score'] > answer_finetuned['score']:
    answer =  answer_dict['answer']
else:
    answer = answer_finetuned['score']
    print(answer)
    #question in french
    print(french_translator(question, max_length=40))
    #answer in french
    print(french_translator(answer, max_length=40))
    #question in german
    print(german_translator(question, max_length=40))
    #answer in german
    print(german_translator(answer, max_length =40))