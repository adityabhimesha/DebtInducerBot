# pragma pylint: disable=missing-docstring, invalid-name, pointless-string-statement
# flake8: noqa: F401
# isort: skip_file
# --- Do not remove these libs ---
import numpy as np  # noqa
import pandas as pd  # noqa
from pandas import DataFrame
import logging

logger = logging.getLogger(__name__)

from freqtrade.strategy import (BooleanParameter, CategoricalParameter, DecimalParameter,
                                IStrategy, IntParameter)

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

from DQN import Agent
from utils import *

# This class is a sample. Feel free to customize it.
class DeepQStrategy(IStrategy):

    window_size = 10+3 

    agent = Agent(window_size, True)


    # Strategy interface version - allow new iterations of the strategy interface.
    # Check the documentation or the Sample strategy to get the latest version.
    INTERFACE_VERSION = 3

    # Can this strategy go short?
    can_short: bool = False

    # Minimal ROI designed for the strategy.
    # This attribute will be overridden if the config file contains "minimal_roi".
    minimal_roi = {
        "0": 0.02
    }

    # Optimal stoploss designed for the strategy.
    # This attribute will be overridden if the config file contains "stoploss".
    stoploss = -0.02

    # Trailing stoploss
    trailing_stop = False
    # trailing_only_offset_is_reached = False
    #trailing_stop_positive = 0.02
    # trailing_stop_positive_offset = 0.0  # Disabled / not configured

    # Optimal timeframe for the strategy.
    timeframe = '1m'

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the config.
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False


    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 30

    # Optional order type mapping.
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'market',
        'stoploss_on_exchange': False
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    plot_config = {

    }

    def informative_pairs(self):
        return []

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        dataframe['enter_short'] = 0
        dataframe['enter_long'] = 0
       

        for i in range(0, dataframe.shape[0]):
            # logger.info(dataframe.iloc[:, dataframe.columns.get_loc('close')])
            if(i > self.window_size):
                state = generate_combined_state(i, self.window_size, dataframe.iloc[:, dataframe.columns.get_loc('close')])

                action = np.argmax(self.agent.model.predict(state, verbose = 0)[0])
                
                if action == 1: #Buy
                    print("Buy Taken")
                    dataframe.iloc[i, dataframe.columns.get_loc('enter_long')] = 1
                if action == 2: #Sell
                    pass


        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_short'] = 0
        dataframe['exit_long'] = 0

        return dataframe
