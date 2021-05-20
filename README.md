# Assignment 1 - mgmt590-lec1

**Project Description:**
Question Answering has become one of the most important problems in modern NLP research. In this project, I have tried to implement a better model which would be better for answering questions.

**The following changes have been made in the code:**

**Updated Model used:** 
*bert-small-finetuned-squadv2-* The smaller BERT models are most effective in the context of knowledge distillation.
To do well on SQuAD2.0, systems must not only answer questions when possible, but also determine when no answer is supported by the paragraph and abstain from answering.
Due to the above reasons, this model would give better solutions.

**Updated print statements:**
I have added question number specific print statements to answer questions more systematically. 

**Added comments:**
I have added comments for better representation of the code
