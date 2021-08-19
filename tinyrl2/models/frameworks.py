"""RL algorithm frameworks
"""
import gym

from tinyrl2.interfaces import PolicyValueModel, ValueModel


class ActorCriticModel(PolicyValueModel):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            policy,
            value,
    ):
        super().__init__(observation_space, action_space)
        self.policy = policy
        self.value = value


class DDPGModel(PolicyValueModel):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            policy,
            value,
    ):
        super().__init__(observation_space, action_space)
        self.policy = policy
        self.value = value

    def forward(self, obs):
        pass

    def forward_policy(self, obs):
        gradient = self.value td error something

        something = self.policy(obs)

        policy = td error + something

        return policy


class DQN(ValueModel):
    def __init__(self):
        pass
