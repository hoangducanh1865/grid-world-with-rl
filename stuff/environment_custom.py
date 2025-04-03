'''
Custom Python Class
'''

# Check test.py for example

class BaseEnvironment:
    
    def __init__(self):
        self.reward = None
        self.state = None
        self.termination = None
        self.reward_state_term = (self.reward, self.state, self.termination)

    def env_init(self, env_info={}):
        pass
    
    def env_start(self):
        pass
    
    def env_step(self, action):
        pass
    