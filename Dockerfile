FROM tensorflow/tensorflow

COPY requirements.txt . 

RUN pip install -r requirements.txt 

COPY Answer.py .

CMD ["python3", "Answer.py"]
