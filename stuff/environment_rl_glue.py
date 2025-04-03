'''
Use RL-Glue
'''

from __future__ import print_function

from abc import ABCMeta, abstractmethod

class BaseEnvironment:
    __metaclass__ = ABCMeta
    
    def __init__(self):
        reward = None
        state = None
        termination = None
        self.reward_state_term = (reward, state, termination)
    
    @abstractmethod
    def env_init(self, env_info={}):
        pass
    
    @abstractmethod
    def env_start(self):
        pass
    
    @abstractmethod
    def env_step(self, action):
        pass
