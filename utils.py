import pandas as pd

from constants import HANDLING_COSTS, MWST


def get_possible_prices(ek, uvp):
    prices = []
    possible_prices = pd.read_csv("assets/possible_prices.csv", sep=";",
                                  header=None,
                                  names=["prices"])
    # add vat + handling costs
    ek = (ek * (100 + HANDLING_COSTS) / 100) * (100 + MWST) / 100
    counter = 0
    while ek > possible_prices['prices'][counter]:
        counter += 1
    prices.append(possible_prices['prices'][counter])
    counter += 1
    while possible_prices['prices'][counter] <= uvp:
        prices.append(possible_prices['prices'][counter])
        counter += 1
    return prices


def psychopreiser(price):
    possible_prices = pd.read_csv("assets/possible_prices.csv", sep=",",
                                  header=None,
                                  names=["prices"])
    counter = 0
    while price > possible_prices['prices'][counter]:
        counter += 1
    diff_upper = possible_prices['prices'][counter] - price
    diff_lower = price - possible_prices['prices'][counter-1]
    if diff_lower < diff_upper:
        psychopreis = possible_prices['prices'][counter-1]
    else:
        psychopreis = possible_prices['prices'][counter]
    return psychopreis
