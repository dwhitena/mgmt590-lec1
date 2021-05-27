from transformers.pipelines import pipeline
from flask import Flask
from flask import request
from flask import jsonify
import sqlite3
import time
import os

conn = sqlite3.connect('pythonsqlite.db')
cursor = conn.cursor()
cursor.execute("DROP TABLE IF EXISTS Models")
conn.commit()

#  Creating table as per requirement
sql = '''CREATE TABLE Models
        (name varchar(100), tokenizer varchar(100), model varchar(100));'''
cursor.execute(sql)
print("Table Created Successfully")
cursor.execute('''INSERT INTO Models VALUES("distilled-bert","distilbert-base-uncased-distilled-squad","distilbert-base-uncased-distilled-squad");''')
cursor.execute('''INSERT INTO Models VALUES("deepset-roberta","deepset/roberta-base-squad2","deepset/roberta-base-squad2");''')
conn.commit()
conn.close()

# Create my flask app
app = Flask(__name__)

@app.route("/models", methods=['GET', 'PUT', 'DELETE'])
def models():
    if request.method == 'GET':
        conn = sqlite3.connect('pythonsqlite.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM Models")
        models = cursor.fetchall()
        listmodels = []
        for i in models:
            output = {
                "name": i[0],
                "tokenizer": i[1],
                "model": i[2]
            }
            listmodels.append(output)
        conn.close()
        return jsonify(listmodels)

    elif request.method == 'PUT':
        conn = sqlite3.connect('pythonsqlite.db')
        cursor = conn.cursor()

        insertmodel = request.json
        name = insertmodel['name']
        tokenizer = insertmodel['tokenizer']
        model = insertmodel['model']

        cursor.execute("INSERT INTO Models VALUES (?, ?, ?)", (name, tokenizer, model))
        conn.commit()
        cursor.execute("SELECT * FROM Models")
        models = cursor.fetchall()
        listmodels = []
        for i in models:
            output = {
                "name": i[0],
                "tokenizer": i[1],
                "model": i[2]
            }
            listmodels.append(output)
        conn.close()
        return jsonify(listmodels)

    elif request.method == 'DELETE':
        deletemodel = request.args.get('model')

        conn = sqlite3.connect("pythonsqlite.db")
        c = conn.cursor()
        c.execute("DELETE FROM Models WHERE name = ?", (deletemodel,))
        conn.commit()
        c.execute("SELECT * FROM Models")
        model_all = c.fetchall()
        listmodels = []
        for i in model_all:
            output = {
                "name": i[0],
                "tokenizer": i[1],
                "model": i[2]
            }
            listmodels.append(output)
        conn.close()
        return jsonify(listmodels)



#  Q4
# get answer and log api
@app.route("/answer", methods = ['POST'])
def answer():

    # Get the request body data
    model_name = request.args.get('model', None)
    data = request.json
    if not model_name:
        model_name='distilled-bert'

    # connecting to database
    conn = sqlite3.connect('pythonsqlite.db')
    c = conn.cursor()


    #fetching model details
    c.execute("SELECT DISTINCT name,tokenizer,model FROM models WHERE name=?",(model_name,))
    model_details = c.fetchall()

    row= model_details[0]
    name = row[0]
    tokenizer = row[1]
    model = row[2]

    #loading and implementing model
    hg_comp = pipeline('question-answering', model=model, tokenizer=tokenizer)

    # Answer the answer
    answer = hg_comp({'question': data['question'], 'context': data['context']})['answer']

    #timestamp
    ts = time.time()

    #logging q/a entry in the databse
    c.execute("CREATE TABLE IF NOT EXISTS qa_log(question TEXT, context TEXT, answer TEXT, model TEXT,timestamp REAL)")
    c.execute("INSERT INTO qa_log VALUES(?,?,?,?,?)", (data['question'], data['context'],answer, model_name,ts))
    conn.commit()


    c.close()
    conn.close()

    # Create the response body.
    out = {
        "timestamp": ts,
        "model": model_name,
        "answer": answer,
        "question": data['question'],
        "context": data['context']

    }
    return jsonify(out)


#  #  Question 5  #  #

# recent entries from log

@app.route("/answer1", methods = ['GET'])
def getrecent1():

    #connecting to database
    conn = sqlite3.connect('pythonsqlite.db')
    c = conn.cursor()

    #extracting arguments
    model_name = request.args.get('model', None)
    start = float(request.args.get('start', None))
    end = float(request.args.get('end',None))

    c.execute('SELECT * FROM qa_log WHERE model=? and timestamp between ? and ?',  (model_name, start, end))
    #c.execute('SELECT * FROM qa_log WHERE model=?', (model_name,))

    logdata = c.fetchall()

    recent_list = []
    for row in logdata:

        log_q = row[0]
        log_ct = row[1]
        log_ans = row[2]
        log_mod = row[3]
        log_ts = row[4]

        row_model = {
                "question": log_q,
                "context": log_ct,
                "ans": log_ans,
                "model": log_mod,
                "timestamp": log_ts
            }

        print(row_model)
        recent_list.append(row_model)

    c.close()
    conn.close()
    return jsonify(recent_list)


# Define a handler for the / path, which
# returns "Hello World"
@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


# Run if running "python answer.py"
if __name__ == '__main__':
    # Run our Flask app and start listening for requests!
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 8080)), threaded=True)

conn.close()
