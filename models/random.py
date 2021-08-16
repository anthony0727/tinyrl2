import gym

from interfaces import TorchModel


class RandomModel(TorchModel):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
    ):
        super().__init__(observation_space, action_space)

    def forward(self, obs):
        action = self.action_space.sample()

        return action
