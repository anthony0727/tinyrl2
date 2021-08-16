from .interfaces import Agent


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
