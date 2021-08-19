import gym

from tinyrl2.interfaces import TorchModel, Agent


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


class RandomAgent(Agent):
    def __init__(
            self,
            model,
    ):
        super(RandomAgent, self).__init__(model, num_envs=0)

    def act(self, obs):
        action = self.model(obs)

        return action

    def collect(self):
        pass
