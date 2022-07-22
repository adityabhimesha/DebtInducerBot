import numpy as np
import logging
logger = logging.getLogger(__name__)


        
def sigmoid(x):
    return 1 / (1 + np.exp(-x))



def generate_price_state(end_index, window_size, stock_prices):
    '''
    returns prices smushed by sigmoid 0 to 1 of all window size number of prices
    '''
    start_index = end_index - window_size
    if start_index >= 0:
        period = stock_prices[start_index:end_index+1]
    else: # if end_index cannot suffice window_size, pad with prices on start_index
        period = -start_index * [stock_prices[0]] + stock_prices[0:end_index+1]

    return sigmoid(np.diff(period))



def generate_combined_state(end_index, window_size, stock_prices):
    '''
    returns smushed value of all <window size> prices + portfolio state(price, balance, inventory)
    '''
    price_state = generate_price_state(end_index, window_size, stock_prices.to_numpy())
    result = np.array([np.concatenate((price_state), axis=None)])
    # logger.info(result)

    return result

