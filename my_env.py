import numpy as np
import pandas as pd
from gym.utils import seeding
import gym
from gym import spaces
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3.common.vec_env import DummyVecEnv
from pypfopt import EfficientFrontier, objective_functions
from pypfopt import risk_models
from pypfopt import expected_returns

class StockPortfolioEnv(gym.Env):
    """A single stock trading environment for OpenAI gym

    Attributes
    ----------
        df: DataFrame
            input data
        stock_dim : int
            number of unique stocks
        hmax : int
            maximum number of shares to trade
        initial_amount : int
            start money
        transaction_cost_pct: float
            transaction cost percentage per trade
        reward_scaling: float
            scaling factor for reward, good for training
        state_space: int
            the dimension of input features
        action_space: int
            equals stock dimension
        tech_indicator_list: list
            a list of technical indicator names
        turbulence_threshold: int
            a threshold to control risk aversion
        day: int
            an increment number to control date

    Methods
    -------
    _sell_stock()
        perform sell action based on the sign of the action
    _buy_stock()
        perform buy action based on the sign of the action
    step()
        at each step the agent will return actions, then 
        we will calculate the reward, and return the next observation.
    reset()
        reset the environment
    render()
        use render to return other functions
    save_asset_memory()
        return account value at each time step
    save_action_memory()
        return actions/positions at each time step
        

    """
    metadata = {'render.modes': ['human']}


    def __init__(self, 
                df,
                stock_dim,
                hmax,
                initial_amount,
                transaction_cost_pct,
                reward_scaling,
                state_space,
                action_space,
                tech_indicator_list,
                initial_weights,
                turbulence_threshold=None,
                lookback=252,
                day = 0):
        #super(StockEnv, self).__init__()
        #money = 10 , scope = 1
        self.day = day
        self.lookback=lookback
        self.df = df
        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.transaction_cost_pct =transaction_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        
        self.tech_indicator_list = tech_indicator_list
        self.initial_weights = initial_weights

        # action_space normalization and shape is self.stock_dim
        self.action_space = spaces.Box(low = 0, high = 1,shape = (10,)) 
 
        self.observation_space = spaces.Box(
        low=-np.inf,  # Minimum value for each element
        high=np.inf,  # Maximum value for each element
        shape=(10,10),  # Rows = tickers, Columns = features
        dtype=np.float64  # Data type for the observation
        )

        # load data from a pandas dataframe
        unique_dates = self.df['date'].unique()
        current_date = unique_dates[self.day]
        self.data = self.df[self.df['date'] == current_date]
        
        self.state = np.array([self.data[tech] for tech in self.tech_indicator_list]).T

        self.terminal = False     
        self.turbulence_threshold = turbulence_threshold        
        # initalize state: inital portfolio return + individual stock return + individual weights
        self.portfolio_value = self.initial_amount
    
        # memorize portfolio value each step
        self.asset_memory = [self.initial_amount]
        # memorize portfolio return each step
        self.portfolio_return_memory = [0]
        self.actions_memory=[self.initial_weights]
        self.date_memory = [unique_dates[self.day]]

             
    def step(self, actions):
        # print(self.day)
        self.terminal = self.day >= len(self.df.index.unique())-1
        #print("actions:",actions)

        if self.terminal:
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']
           # plt.plot(df.daily_return.cumsum(),'r')
          #  plt.savefig('results/cumulative_reward.png')
           # plt.close()
            
            #plt.plot(self.portfolio_return_memory,'r')
            #plt.savefig('results/rewards.png')
            #plt.close()

            #print("=================================")
            #print("begin_total_asset:{}".format(self.asset_memory[0]))           
            #print("end_total_asset:{}".format(self.portfolio_value))

            #df_daily_return = pd.DataFrame(self.portfolio_return_memory)
            #df_daily_return.columns = ['daily_return']
            #if df_daily_return['daily_return'].std() !=0:
            #  sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/ \
            #           df_daily_return['daily_return'].std()
            #  print("Sharpe: ",sharpe)
            #print("=================================")
            
            return self.state, self.reward, self.terminal,{}

        else:
            #  norm_actions = actions
            weights = self.l1_normalization(actions)
            #weights = self.softmax_normalization(actions) 
            #print("Normalized actions: ", weights)
            self.actions_memory.append(weights)
            last_day_memory = self.data.close
        

            """
            # Get data frame of close prices 
            # Reset the Index to tic and date
            df_prices = self.data.copy()
            df_prices = df_prices.reset_index().set_index(['tic', 'date']).sort_index()
            tic_list = list(set([i for i,j in df_prices.index]))

            # Get all the Close Prices
            df_close = pd.DataFrame()
            for ticker in tic_list:
                series = df_prices.xs(ticker).close
                df_close[ticker] = series
            
            #mu = expected_returns.mean_historical_return(df_close)
            Sigma = risk_models.sample_cov(df_close)
            ef = EfficientFrontier(mu,Sigma)

            raw_weights = ef.max_sharpe()
            weights = [j for i,j in raw_weights.items()]
            self.actions_memory.append(weights)
            last_day_memory = self.data
            
            """
		
            #load next state
            # Load next state
        # Load next state
        self.day += 1
        unique_dates = self.df['date'].unique()

        # Check if self.day exceeds the length of unique_dates
        if self.day >= len(unique_dates):
            #self.day = 0  # Reset to the first date (or set to a termination condition)
            self.terminal=True
            
            df = pd.DataFrame(self.portfolio_return_memory)
            df.columns = ['daily_return']
           # plt.plot(df.daily_return.cumsum(),'r')
           # plt.savefig('results/cumulative_reward.png')
           # plt.close()
            
           # plt.plot(self.portfolio_return_memory,'r')
         #   plt.savefig('results/rewards.png')
        #    plt.close()

           # print("=================================")
           # print("begin_total_asset:{}".format(self.asset_memory[0]))           
          #  print("end_total_asset:{}".format(self.portfolio_value))

           # df_daily_return = pd.DataFrame(self.portfolio_return_memory)
           # df_daily_return.columns = ['daily_return']
        #    if df_daily_return['daily_return'].std() !=0:
           #   sharpe = (252**0.5)*df_daily_return['daily_return'].mean()/ \
           #            df_daily_return['daily_return'].std()
           #   print("Sharpe: ",sharpe)
           # print("=================================")
            
            return self.state, self.reward, self.terminal,{}
        
        else:
            pass

        current_date = unique_dates[self.day]
        self.data = self.df[self.df['date'] == current_date]

        # Update state with all the indicators indicators (flattened)
        self.state = np.array([self.data[tech] for tech in self.tech_indicator_list]).T

        #Debugging prints to check values
        #print(f"Current Date: {current_date}")
        #print(f"self day: {self.day}")
        #print("Shape of weights:", len(weights))# Assuming weights is a list
        #print("Weights:", weights)

        
        # Ensure that self.data.close.to_numpy() and last_day_memory have the same length
        if len(self.data.close.to_numpy()) != len(last_day_memory.to_numpy()):
            min_length = min(len(self.data.close.to_numpy()), len(last_day_memory.to_numpy()))
            self.data.close = self.data.close.iloc[:min_length]  # Use iloc to slice
            last_day_memory = last_day_memory.iloc[:min_length]  # Use iloc for pandas series slicing

        # Calculate portfolio return: (current close prices / last day's close prices) - 1
        individual_returns = (self.data.close.to_numpy() / last_day_memory.to_numpy() - 1)

        # Ensure weights are properly aligned to the number of stocks (i.e., lengths match)
        portfolio_return = np.dot(individual_returns, np.array(weights))
        
        # Update portfolio value based on the portfolio return
        new_portfolio_value = self.portfolio_value * (1 + portfolio_return)
        self.portfolio_value = new_portfolio_value

        # Save relevant values into memory for later reference
        self.portfolio_return_memory.append(portfolio_return)
        self.date_memory.append(current_date)  # Append the current date
        self.asset_memory.append(new_portfolio_value)

        # Calculate the reward, which could be the updated portfolio value, scaled by a factor
        self.reward = portfolio_return * self.reward_scaling
        #print("Step reward (scaled):", self.reward)

        # Optional debug logging for portfolio return and value updates
        #print(f"New portfolio value: {new_portfolio_value}, Portfolio return: {portfolio_return}")

        # Returning the state, reward, done flag (self.terminal), and additional info
        return self.state, self.reward, self.terminal, {}




    def reset(self):
        self.asset_memory = [self.initial_amount]
        self.day = 0
        unique_dates = self.df['date'].unique()
        current_date = unique_dates[self.day]
        self.data = self.df[self.df['date'] == current_date]
        # load states
        self.state = np.array([self.data[tech] for tech in self.tech_indicator_list]).T
        self.portfolio_value = self.initial_amount
        #self.cost = 0
        #self.trades = 0
        self.terminal = False 
        self.portfolio_return_memory = [0]
              
        self.actions_memory=[self.initial_weights] 
        self.date_memory = [unique_dates[self.day]]  #Correctly reset to the first date

        return self.state
    
    def render(self, mode='human'):
        return self.state
    
    def l1_normalization(self, actions):
        actions = np.maximum(actions, 0)  # Ensure all actions are non-negative
        #print("Raw actions before L1 normalization:", actions)  # Debugging print
        total = np.sum(actions)
        if total > 0:
            return actions / total
        else:
            return np.ones_like(actions) / len(actions)
    
    def save_asset_memory(self):
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        #print(len(date_list))
        df_account_value = pd.DataFrame({'date':date_list,'daily_return':portfolio_return})
        return df_account_value

    def save_action_memory(self):
        # date and close price length must match actions length
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['date']
        
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data.tic
        df_actions.index = df_date.date
        return df_actions
    
    #def initial_weights(self, data_frame):
        # Get data frame of close prices 
        # Reset the Index to tic and date
        #df_prices = data_frame.copy()
        #df_prices = df_prices.reset_index().set_index(['tic', 'date']).sort_index()
        #tic_list = list(set([i for i,j in df_prices.index]))
        
        # Get all the Close Prices
        #df_close = pd.DataFrame()
        #for ticker in tic_list:
            #series = df_prices.xs(ticker).close
            #df_close[ticker] = series
            
        #mu = expected_returns.mean_historical_return(df_close)
        #Sigma = risk_models.sample_cov(df_close)
        #ef = EfficientFrontier(mu,Sigma, weight_bounds=(0.01, 1))
        
        #raw_weights = ef.max_sharpe()
        #initial_weights = [j for i,j in raw_weights.items()]
        #num_stocks = len(data_frame['tic'].unique())
        #initial_weights = [1.0 / num_stocks] * num_stocks
        #return initial_weights

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs