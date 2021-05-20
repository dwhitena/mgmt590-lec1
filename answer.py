import logging
import pprint as pp
import re
from collections import OrderedDict
import pandas as pd
import torch
import wikipedia as wiki
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from transformers import pipeline

model_names = ["deepset/roberta-base-squad2", "bert-large-uncased-whole-word-masking-finetuned-squad",
               "ahotrod/electra_large_discriminator_squad2_512"]

# Suppress irrelevant warning
# To control logging level for various modules used in the application:
def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{"|".join(prefices)})')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


set_global_logging_level(logging.ERROR)

df = pd.read_csv('train-squad.csv')
df = df.iloc[0:10]

contexts = df['context'].to_list()
questions = df['question'].to_list()

print(f"Predicting Answers to {len(questions)} Questions using {len(model_names)} different models...\n")

predicted_answers = [[0 for x in range(len(model_names))] for y in range(len(questions))]
predicted_scores = [[0 for x in range(len(model_names))] for y in range(len(questions))]

for model_index in range(len(model_names)):
    # Retrieve the pre-trained model. Slow.
    nlp = pipeline('question-answering', model=model_names[model_index], tokenizer=model_names[model_index])

    question_index = 0
    for question, context in zip(questions, contexts):
        # Predict the answer
        answer = nlp(question=question, context=context)
        predicted_answers[question_index][model_index] = answer['answer']
        predicted_scores[question_index][model_index] = answer['score']
        question_index += 1

for question_index in range(0, len(questions)):
    print(f'Question: {questions[question_index]}')
    for model_index in range(0, len(model_names)):
        print(f"Model: {model_names[model_index]} Answer: '{predicted_answers[question_index][model_index]}'")
    print("\n")


# executing these commands for the first time initiates a download of the
# model weights to ~/.cache/torch/transformers/
tokenizer = AutoTokenizer.from_pretrained("deepset/bert-base-cased-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/bert-base-cased-squad2")

question = 'What is the wingspan of an albatross?'

results = wiki.search(question)
print("Wikipedia search results for our question:\n")
pp.pprint(results)

page = wiki.page(results[0])
text = page.content
print(f"\nThe {results[0]} Wikipedia article contains {len(text)} characters.")

inputs = tokenizer.encode_plus(question, text, return_tensors='pt')
print(f"This translates into {len(inputs['input_ids'][0])} tokens.")

# time to chunk!
# identify question tokens (token_type_ids = 0)
qmask = inputs['token_type_ids'].lt(1)
qt = torch.masked_select(inputs['input_ids'], qmask)
print(f"The question consists of {qt.size()[0]} tokens.")

chunk_size = model.config.max_position_embeddings - qt.size()[0] - 1  # the "-1" accounts for
# having to add a [SEP] token to the end of each chunk
print(f"Each chunk will contain {chunk_size - 2} tokens of the Wikipedia article.")

# create a dict of dicts; each sub-dict mimics the structure of pre-chunked model input
chunked_input = OrderedDict()
for k, v in inputs.items():
    q = torch.masked_select(v, qmask)
    c = torch.masked_select(v, ~qmask)
    chunks = torch.split(c, chunk_size)

    for i, chunk in enumerate(chunks):
        if i not in chunked_input:
            chunked_input[i] = {}

        thing = torch.cat((q, chunk))
        if i != len(chunks) - 1:
            if k == 'input_ids':
                thing = torch.cat((thing, torch.tensor([102])))
            else:
                thing = torch.cat((thing, torch.tensor([1])))

        chunked_input[i][k] = torch.unsqueeze(thing, dim=0)

for i in range(len(chunked_input.keys())):
    print(f"Number of tokens in chunk {i}: {len(chunked_input[i]['input_ids'].tolist()[0])}")

def convert_ids_to_string(tokenizer, input_ids):
    return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids))

answer = ''

# now we iterate over our chunks, looking for the best answer from each chunk
for _, chunk in chunked_input.items():
    a = model(**chunk)
    answer_start_scores = a[0]
    answer_end_scores = a[1]
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1
    ans = convert_ids_to_string(tokenizer, chunk['input_ids'][0][answer_start:answer_end])
    # if the ans == [CLS] then the model did not find a real answer in this chunk
    if ans != '[CLS]':
        answer += ans + " / "

print(answer)

class DocumentReader:
    def __init__(self, pretrained_model_name_or_path='bert-large-uncased'):
        self.READER_PATH = pretrained_model_name_or_path
        self.tokenizer = AutoTokenizer.from_pretrained(self.READER_PATH)
        self.model = AutoModelForQuestionAnswering.from_pretrained(self.READER_PATH)
        self.max_len = self.model.config.max_position_embeddings
        self.chunked = False

    def tokenize(self, question, text):
        self.inputs = self.tokenizer.encode_plus(question, text, add_special_tokens=True, return_tensors="pt")
        self.input_ids = self.inputs["input_ids"].tolist()[0]

        if len(self.input_ids) > self.max_len:
            self.inputs = self.chunkify()
            self.chunked = True

    def chunkify(self):
        """
        Break up a long article into chunks that fit within the max token
        requirement for that Transformer model.

        Calls to BERT / RoBERTa / ALBERT require the following format:
        [CLS] question tokens [SEP] context tokens [SEP].
        """

        # create question mask based on token_type_ids
        # value is 0 for question tokens, 1 for context tokens
        qmask = self.inputs['token_type_ids'].lt(1)
        qt = torch.masked_select(self.inputs['input_ids'], qmask)
        chunk_size = self.max_len - qt.size()[0] - 1  # the "-1" accounts for
        # having to add an ending [SEP] token to the end

        # create a dict of dicts; each sub-dict mimics the structure of pre-chunked model input
        chunked_input = OrderedDict()
        for k, v in self.inputs.items():
            q = torch.masked_select(v, qmask)
            c = torch.masked_select(v, ~qmask)
            chunks = torch.split(c, chunk_size)

            for i, chunk in enumerate(chunks):
                if i not in chunked_input:
                    chunked_input[i] = {}

                thing = torch.cat((q, chunk))
                if i != len(chunks) - 1:
                    if k == 'input_ids':
                        thing = torch.cat((thing, torch.tensor([102])))
                    else:
                        thing = torch.cat((thing, torch.tensor([1])))

                chunked_input[i][k] = torch.unsqueeze(thing, dim=0)
        return chunked_input

    def get_answer(self):
        if self.chunked:
            answer = ''
            for k, chunk in self.inputs.items():
                answer_start_scores, answer_end_scores = self.model(**chunk, return_dict=False)

                answer_start = torch.argmax(answer_start_scores)
                answer_end = torch.argmax(answer_end_scores) + 1

                ans = self.convert_ids_to_string(chunk['input_ids'][0][answer_start:answer_end])
                if ans != '[CLS]':
                    answer += ans + " / "
            return answer
        else:
            answer_start_scores, answer_end_scores = self.model(**self.inputs)

            answer_start = torch.argmax(
                answer_start_scores)  # get the most likely beginning of answer with the argmax of the score
            answer_end = torch.argmax(
                answer_end_scores) + 1  # get the most likely end of answer with the argmax of the score

            return self.convert_ids_to_string(self.inputs['input_ids'][0][
                                              answer_start:answer_end])
    def convert_ids_to_string(self, input_ids):
        return self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids))

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)

questions = [
    'Why is the sky blue?',
    'How many sides does a pentagon have?'
]

reader = DocumentReader("deepset/bert-base-cased-squad2")

# if you trained your own model using the training cell earlier, you can access it with this:
# reader = DocumentReader("./models/bert/bbu_squad2")

for question in questions:
    print(f"Question: {question}")
    results = wiki.search(question)
    page = wiki.page(results[0])
    print(f"Top wiki result: {page}")
    text = page.content
    reader.tokenize(question, text)
    print(f"Answer: {reader.get_answer()}")
    print()






