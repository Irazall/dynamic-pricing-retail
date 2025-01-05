import pickle
from datetime import datetime, timedelta
from typing import Dict

from flask_pymongo import PyMongo

from constants import NEW_DAYS_FOR_NEXT_TRAIN

mongodb_client = PyMongo()


def db_save_models(ai_pricer):
    main_db = mongodb_client.cx['main']
    main_db.overview.update_one({'aid': ai_pricer.aid},
                                {"$set": {'aid': ai_pricer.aid,
                                 "last_trained": ai_pricer.last_trained,
                                 "last_day_for_training": ai_pricer.last_day_for_training,
                                 "sufficient_data": ai_pricer.sufficient_data}},
                                upsert=True)
    main_db.models.remove({'aid': ai_pricer.aid})
    for i in ai_pricer.models:
        dump = pickle.dumps(ai_pricer.models[i])
        main_db.models.insert({'aid': ai_pricer.aid, "price": i, 'trace': dump})
    if ai_pricer.models:
        print(f'Wrote models for article {ai_pricer.aid}...')


def db_load_models(aid) -> Dict:
    models = {}
    results = mongodb_client.cx['main'].models.find({'aid': aid})
    for doc in results:
        models.update({doc['price']: pickle.loads(doc['trace'])})
    return models


def db_read_overview(aid) -> Dict:
    result = mongodb_client.cx['main'].overview.find_one({'aid': aid})
    return result


def db_save_input(df, curr_index):
    input_db = mongodb_client.cx['input']
    df.reset_index(inplace=True)
    data_dict = df.to_dict("records")
    input_db.bestelldaten.insert_many(data_dict)
    print(f'Wrote input from index {curr_index}...')


def db_get_highest_index_input():
    input_db = mongodb_client.cx['input']
    try:
        result = input_db.bestelldaten.find().sort('index', -1).limit(1)[0]
        return result['index']
    except IndexError:
        return 3000000


def db_get_trainable_articles():
    main_db = mongodb_client.cx['main']
    results = main_db.overview.find(
        {'$or': [
            {'last_day_for_training': {'$lt': datetime.today() - timedelta(days=NEW_DAYS_FOR_NEXT_TRAIN)}},
            {'last_day_for_training': 'null'}
        ]})
    response = {}
    for doc in results:
        response.update({'aid': doc['aid'],
                         'last_day_for_training': doc['last_day_for_training'],
                         'last_trained': doc['last_trained']})
    return response
