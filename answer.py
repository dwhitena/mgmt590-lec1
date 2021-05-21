import pandas as pd
from transformers.pipelines import pipeline

hg_comp =  pipeline('question-answering', model="distilbert-base-uncased-distilled-squad", tokenizer="distilbert-base-uncased-distilled-squad")
hg_comp_finetuned = pipeline('question-answering', model="bert-large-uncased-whole-word-masking-finetuned-squad", tokenizer="bert-large-uncased-whole-word-masking-finetuned-squad")
translator_french = pipeline("translation_en_to_fr")
translator_german = pipeline("translation_en_to_de")

data = pd.read_csv('examples.csv')

for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    answer_dict = hg_comp({'question': question, 'context': context})
    answer_finetuned = hg_comp_finetuned({'question': question, 'context': context})

answer =''
    if answer_dict['score'] > answer_finetuned['score']:
        answer =  answer_dict['answer']
    else:
        answer = answer_finetuned['score']
    print(answer)
    #question in french
    print(translator_french(question, max_length=40))
    #answer in french
    print(translator_french(answer, max_length=40))
    #question in german
    print(translator_german(question, max_length=40))
    #answer in german
    print(translator_german(answer, max_length =40))

#using batch processing i.e. processing mutliple questions together and getting answer for them as a batch
from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased',return_token_type_ids = True)
model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')

question_context_for_batch = []

for idx, row in data.iterrows():
    context = row['context']
    question = row['question']
    question_context_for_batch.append((question, context))

encoding = tokenizer.batch_encode_plus(question_context_for_batch,pad_to_max_length=True, return_tensors="pt")
input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
output = model(input_ids, attention_mask=attention_mask)

for index,(start_score,end_score,input_id) in enumerate(zip(output.start_logits,output.end_logits,input_ids)):
    max_startscore = torch.argmax(start_score)
    max_endscore = torch.argmax(end_score)
    ans_tokens = input_ids[index][max_startscore: max_endscore + 1]
    answer_tokens = tokenizer.convert_ids_to_tokens(ans_tokens, skip_special_tokens=True)
    answer_tokens_to_string = tokenizer.convert_tokens_to_string(answer_tokens)
    print ("\nQuestion: ",questions[index])
    print ("Answer: ", answer_tokens_to_string)
    #Printing Questions and Answers in French and German
    # question in french
    print(translator_french(questions[index], max_length=40))
    # answer in french
    print(translator_french(answer_tokens_to_string, max_length=40))
    # question in german
    print(translator_german(questions[index], max_length=40))
    # answer in german
    print(translator_german(answer_tokens_to_string, max_length=40))
