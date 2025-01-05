import random
import sys
from datetime import datetime
from typing import Tuple, Dict

import pandas as pd
import numpy as np
import pymc3 as pm

from constants import MIN_OBS_PER_PRICE, MWST, MAX_PRICE_VAR, MIN_STD_MODELS, \
    MIN_DIFF_PRICES, HANDLING_COSTS
from db import db_save_models, db_load_models, mongodb_client, db_read_overview
from utils import get_possible_prices


class AiPricer:

    def __init__(self, aid, uvp,
                 min_obs_per_price=MIN_OBS_PER_PRICE,
                 min_diff_prices=MIN_DIFF_PRICES,
                 min_std_models=MIN_STD_MODELS,
                 max_price_var=MAX_PRICE_VAR,
                 mwst=MWST):
        self.aid = str(aid)
        self.uvp = float(uvp)
        self._max_ek = None
        self._data = None
        self._prices = None
        self.last_trained = None
        self.last_day_for_training = None
        self.min_obs_per_price = min_obs_per_price
        self.min_diff_prices = min_diff_prices
        self.min_std_models = min_std_models
        self.max_price_var = max_price_var  # one more needed
        self.models = {}
        self.mwst = mwst  # in percent
        self._sufficient_data = None
        try:
            self.trained = db_read_overview(self.aid)['last_trained'] is not None
        except TypeError:
            self.trained = False

    @property
    def data(self):
        if self._data is None:
            input_db = mongodb_client.cx['input']
            bestelldaten = input_db.bestelldaten
            self._data = pd.DataFrame(list(bestelldaten.find(
                {'aid': int(self.aid)})))
            if len(self._data) > 0:
                # delete sellingPrice == 0
                self._data = self._data[self._data['sellingPrice'] != 0]
                # aggregate to one day
                # TODO keep only highest quantity if more than one price per day
                self._data = self._data.groupby('dateOfOrder', as_index=False).agg(
                    {'quantity': sum,
                     'sellingPrice': max,
                     'netSellingPrice': max,
                     'vat': max,
                     'buyingPrice': max})
                # add zero days
                self._data['dateOfOrder'] = pd.to_datetime(self._data['dateOfOrder'],
                                                           format='%Y-%m-%d')
                self._data = self._data.set_index('dateOfOrder').sort_values('dateOfOrder')
                self._data = self._data.asfreq('D')
                self._data['sellingPrice'] = self._data['sellingPrice'].fillna(method='ffill')
                self._data['quantity'] = self._data['quantity'].fillna(0)
                self._data['dateOfOrder'] = self._data.index
                self._data.reset_index(drop=True)
        return self._data

    @property
    def max_ek(self):
        if self._max_ek is None:
            self._max_ek = min(self.data['buyingPrice'])
        return self._max_ek

    @property
    def prices(self):
        if self._prices is None:
            self._prices = get_possible_prices(self.max_ek, self.uvp)
        return self._prices

    @property
    def sufficient_data(self):
        if self._sufficient_data is None:
            if len(self.data) == 0:
                self._sufficient_data = False
            else:
                counter_sufficient = 0
                for price in self.prices:
                    amount_data_per_price = len(self.data[self.data['sellingPrice'] ==
                                                          price])
                    if amount_data_per_price >= self.min_obs_per_price:
                        counter_sufficient += 1
                if counter_sufficient >= self.min_diff_prices:
                    self._sufficient_data = True
                else:
                    self._sufficient_data = False
        return self._sufficient_data

    ####################
    # Functions
    ####################

    def calculate_best_price(self) -> Dict:
        test_price = not self.sufficient_data
        if test_price:
            best_price = self.get_next_price(
                self.get_last_price(),
                random.choice([i for i in list(self.max_price_var) if i != 0])
            )
            exp_demand = None
            exp_costs_per_unit = None
            exp_profit_per_unit = None
        else:
            if not self.trained:
                return {
                    'message': 'Train model!'
                }
            else:
                self.models = db_load_models(self.aid)
            index = self.prices.index(self.get_last_price())
            possible_prices = []
            for i in self.max_price_var:
                try:
                    possible_prices.append(self.prices[index + i])
                except IndexError:
                    pass
            demands = self.sample_demands_from_model(possible_prices)
            best_price, exp_demand, exp_costs, exp_profit = \
                self.optimal_price(possible_prices, demands)
            exp_demand = int(exp_demand)
            if exp_demand > 0:
                exp_costs_per_unit = round(exp_costs/exp_demand, 2)
                exp_profit_per_unit = round(exp_profit/exp_demand, 2)
            else:
                exp_costs_per_unit = 0
                exp_profit_per_unit = 0
        return {
            'best_price': best_price,
            'exp_demand': exp_demand,
            'exp_costs_per_unit': exp_costs_per_unit,
            'exp_profit_per_unit': exp_profit_per_unit,
            'test_price': test_price,
            'model_variance_sufficient': self.model_variance_sufficient()
        }

    def calculate_hk(self, quantity=1):
        return quantity * (self.get_last_price() * HANDLING_COSTS / 100)

    def costs(self, quantity=1):
        costs = round(quantity * (self.calculate_hk(quantity) + self.get_last_ek()), 2)
        return costs

    def get_last_price(self):
        return self.data['sellingPrice'][len(self.data) - 1]

    def get_last_ek(self):
        return self.data['buyingPrice'][len(self.data) - 1]

    def get_next_price(self, price, direction=1) -> float:
        index = self.prices.index(price)
        try:
            next_price = self.prices[index + direction]
        except IndexError:
            next_price = price
        return next_price

    def get_thetas(self) -> Dict:
        if not self.models:
            self.models = db_load_models(self.aid)
        thetas = {}
        for i in self.models:
            thetas.update({str(i): np.mean(self.models[i].theta)})
        return thetas

    def model_variance_sufficient(self) -> bool:
        if not self.trained:
            self.train_models()
        thetas = self.get_thetas()
        # return False if only testprice
        if not thetas:
            return False
        std = np.std(list(thetas.values()))
        if std < self.min_std_models:
            return False
        else:
            return True

    def optimal_price(self, curr_prices, curr_demands) -> Tuple[float, float,
                                                                float, float]:
        curr_costs = []
        for i in curr_demands:
            if i is not None:
                curr_costs.append(self.costs(i))
            else:
                curr_costs.append(0)
        curr_profit = np.subtract(
            np.multiply(np.multiply(curr_prices, (100 - self.mwst)/100),
                        curr_demands),
            curr_costs)
        price_index = np.argmax(curr_profit)
        return (curr_prices[price_index], curr_demands[price_index],
                curr_costs[price_index], curr_profit[price_index])

    def sample_demands_from_model(self, current_prices):
        current_demands = []
        for current_price in current_prices:
            # TODO better to get mean? not following tutorial, but test
            #  nonetheless
            index_price = self.prices.index(current_price)
            if str(current_price) in self.models:
                current_demands.append(
                    np.random.poisson(
                        np.random.choice(self.models[str(current_price)].theta), 1)[0]
                )
            else:
                current_demands.append(0)
        return current_demands

    def train_models(self):
        data_train = self.data
        s_list = {}
        if self.sufficient_data:
            for i in self.prices:
                print(f'Training article {self.aid} for price {i}...',
                      file=sys.stdout)
                data_tmp = data_train[data_train.sellingPrice == i]
                d0 = data_tmp['quantity']  # observed demands (for each offered price)
                if len(d0) > 0:
                    with pm.Model() as m:
                        # priors
                        d = pm.Gamma('theta', 1, 1)  # prior distribution
                        pm.Poisson('d0', d, observed=d0)  # likelihood
                        s = pm.sample(5000, tune=1000)  # inference
                        s_list.update({str(i): s})
            self.trained = True
            self.last_trained = datetime.now()
            self.last_day_for_training = self.data['dateOfOrder'][len(self.data)-1]
            self.models = s_list

        # save either way
        db_save_models(self)






