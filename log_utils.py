import numpy as np
import matplotlib.pyplot as plt
import copy

class RLLogger():
    def __init__(self):
        self.states = []
        self.actions = []
        self.losses_critic = []
        self.losses_actor = []
        self.episode_states = []
        self.episode_actions = []
        self.episode_losses_critic = []
        self.episode_losses_actor = []
        self.episode_steps = []

    def save_state(self, state):
        self.states.append(copy.deepcopy(state))

    def save_episode(self, steps):
        self.episode_states.append(self.states[-steps:])
        self.episode_actions.append(self.actions[-steps:])
        self.episode_losses_critic.append(self.losses_critic[-steps:])
        self.episode_losses_actor.append(self.losses_actor[-steps:])
        self.episode_steps.append(steps)

    def save_loss_critic(self, loss):
        self.losses_critic.append(loss)
    
    def save_loss_actor(self, loss):
        self.losses_actor.append(loss)
    
    def save_action(self, action):
        self.actions.append(action[0])

    

        