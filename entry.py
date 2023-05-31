from flask import Flask
from flask import request
import importlib

from question_embedding import run_question_query

app = Flask(__name__)

@app.route("/answer", methods=['POST'])
def question_embedding():
    data = request.get_json()
    query = data.get('Query')
    answer = run_question_query(query)
    return answer

