# common library
import pandas as pd
import numpy as np
import time
import gym

# RL models from stable-baselines
#from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

from finrl import config
from stable_baselines3 import A2C
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
)

#from stable_baselines3 import SAC


MODELS = {"a2c": A2C}

MODEL_KWARGS = {x: config.__dict__[f"{x.upper()}_PARAMS"] for x in MODELS.keys()}

NOISE = {
    "normal": NormalActionNoise,
    "ornstein_uhlenbeck": OrnsteinUhlenbeckActionNoise,
}

class DRLAgent:
    """Provides implementations for DRL algorithms

    Attributes
    ----------
        env: gym environment class
            user-defined class

    Methods
    -------
    train_A2C()
        the implementation for A2C algorithm
    DRL_prediction()
        make a prediction in a test dataset and get results
    """

    @staticmethod
    def DRL_prediction(model, test_data, test_env, test_obs):
        """make a prediction"""
        start = time.time()
        account_memory = []
        actions_memory = []
        for step_index in range(len(test_data['date'].unique())-1):
            action, _states = model.predict(test_obs)
            test_obs, rewards, dones, info = test_env.step(action)

            # Stop if the environment signals termination
            if dones:
                break
        account_memory = test_env.env_method(method_name="save_asset_memory")
        actions_memory = test_env.env_method(method_name="save_action_memory")
        end = time.time()
        return account_memory[0], actions_memory[0]

    def __init__(self, env):
        self.env = env

    def get_model(
        self,
        model_name,
        policy="MlpPolicy",
        policy_kwargs=None,
        model_kwargs=None,
        verbose=1,
    ):
        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            model_kwargs = MODEL_KWARGS[model_name]

        if "action_noise" in model_kwargs:
            n_actions = self.env.action_space.shape[-1]
            model_kwargs["action_noise"] = NOISE[model_kwargs["action_noise"]](
                mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)
            )
        print(model_kwargs)
        model = MODELS[model_name](
            policy=policy,
            env=self.env,
            tensorboard_log=f"{config.TENSORBOARD_LOG_DIR}/{model_name}",
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            **model_kwargs,
        )
        return model

    #def train_model(self, model, tb_log_name, total_timesteps=5000, log_interval=500):
     #   model = model.learn(
      #      total_timesteps=total_timesteps,
       #     tb_log_name=tb_log_name,
        #    log_interval=log_interval,
         #   callback=monitor_callback
        #)
        #return model
    def train_model(
        self,
        model,
        tb_log_name,
        total_timesteps=5000,
        log_interval=500,
        callback=None
    ):
        """
        Trains the given model using model.learn(), returning the trained model.
        Optionally accepts a custom callback for logging or plotting.

        Parameters
        ----------
        model : A stable-baselines model instance
        tb_log_name : str
            The name under which TensorBoard logs are saved
        total_timesteps : int
            Number of timesteps to train for
        log_interval : int
            Print/log every `log_interval` steps
        callback : BaseCallback or None
            Custom callback for logging, plotting, etc.
        
        Returns
        -------
        model
            The trained model
        """
        model = model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            log_interval=log_interval,
            callback=callback
        )
        return model
