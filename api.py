import threading

import flask
import pandas as pd
from flask import request, jsonify
from bson.json_util import dumps
from bson.json_util import loads

from ai_pricer import AiPricer
from constants import SHOP_ID
from db import mongodb_client, db_save_input, db_get_highest_index_input, \
    db_get_trainable_articles

app = flask.Flask(__name__)
app.config["DEBUG"] = True
app.config["MONGO_URI"] = "mongodb://localhost:27017/main"


mongodb_client.init_app(app)


@app.route('/', methods=['GET'])
def home():
    return f"<h1>Dynamic Pricing</h1><p>This site is a prototype API for the " \
           f"dynamic pricing tool of SO</p>"


# e.g. api/optimal_prices?aids=1,2&uvp=30.99,40.99
@app.route('/api/optimal_prices', methods=['GET'])
def api_optimal_prices():
    if 'aids' in request.args:
        aids = request.args['aids'].split(",")
    else:
        return "Error: No aids field provided. Please specify an aid."
    if 'uvps' in request.args:
        uvps = request.args['uvps'].split(",")
    else:
        return "Error: No uvps field provided. Please specify an uvp."
    response = {}
    for i in range(len(aids)):
        ai = AiPricer(int(aids[i]), float(uvps[i]))
        response.update({str(ai.aid): ai.calculate_best_price()})
    return jsonify(response)


# e.g. api/train_model?aids=1,2&uvp=30.99,40.99
@app.route('/api/train_models', methods=['POST'])
def api_train_models():
    if 'aids' in request.args:
        aids = request.args['aids'].split(",")
    else:
        return "Error: No aids field provided. Please specify an aid."
    if 'uvps' in request.args:
        uvps = request.args['uvps'].split(",")
    else:
        return "Error: No uvps field provided. Please specify an uvp."

    def task(aids_inner, uvps_inner):
        for i in range(len(aids)):
            ai = AiPricer(int(aids_inner[i]), float(uvps_inner[i]))
            ai.train_models()

    thread = threading.Thread(target=task, kwargs={
        'aids_inner': aids, 'uvps_inner': uvps})
    thread.start()
    return {"message": "Thread started"}, 202


# e.g. api/transfer_bestdata
@app.route('/api/transfer_bestdata', methods=['POST'])
def api_put_data():

    def task():
        while True:
            next_index = db_get_highest_index_input() + 1
            url = f"https://XYZ.de/aiTrackerWebhook.php?shopId=" \
                  f"{SHOP_ID}&startIndex={next_index}&quantity=1000"
            df = pd.read_json(url)
            if len(df) != 0:
                df.index += next_index
                db_save_input(df, next_index)
            else:
                break

    thread = threading.Thread(target=task)
    thread.start()
    return {"message": "Thread started"}, 202


# e.g. api/trainable_articles
@app.route('/api/trainable_articles', methods=['GET'])
def api_trainable_articles():
    return jsonify((db_get_trainable_articles()))


app.run()
