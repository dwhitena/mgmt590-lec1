import pandas as pd
from transformers.pipelines import pipeline

data = pd.read_csv('examples.csv')
model_list = ["deepset/roberta-base-squad2", "bert-large-uncased-whole-word-masking-finetuned-squad", "mrm8488/bert-tiny-finetuned-squadv2"]
model_num = len(model_list)
for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    print("The question is", question)
    model_score = {}
    model_answer = {}
    for i in range(model_num):
        print("Predicting answers using model -", model_list[i])
        hg_comp = pipeline(model=model_list[i], tokenizer=model_list[i], task="question-answering")
        answer = hg_comp({'question': question, 'context': context})['answer']
        score = hg_comp({'question': question, 'context': context})['score']
        #print("The answer is",answer)
        #print("The score is",score)
        model_score[model_list[i]] = score
        model_answer[model_list[i]] = answer
    best_model = max(model_score, key=model_score.get)
    print("Best answer for this question was given by model - ",best_model)
    print("Answer with highest confidence score - ",model_answer[best_model])
    print("Confidence score - ",model_score[best_model])