import gym
from torch import nn


# basic modules
class Buffer:
    def __init__(
            self
    ):
        pass


class TorchModel(nn.Module):
    """encapsulates all modules belonging to the implementing rl algorithm
    forward is for used for training, specifically outputs the tensor for loss function
two
    """

    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            deterministic: bool = False,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.deterministic = deterministic

    def forward(self, obs):
        """used to output action

        :param obs: observation
        :return: action
        """
        raise NotImplementedError

    def _test_forward(self):
        sample_obs = self.observation_space.sample()
        sample_output = self(sample_obs)


class PolicyModel(TorchModel):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            deterministic: bool = False,
    ):
        super().__init__(observation_space, action_space, deterministic)

    def forward(self, obs):
        pass


class ValueModel(TorchModel):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            deterministic: bool = False,
    ):
        super().__init__(observation_space, action_space, deterministic)

    def forward(self, obs):
        pass


def _share_encoder(*torch_models):
    encoder = torch_models[0]
    for model in torch_models:
        # TODO: inplace share, resolve double backward if endpoint differs
        model.encoder = encoder


# framework

class PolicyValueModel(TorchModel):
    def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            policy,
            value,
            share_encoder: bool = True,
    ):
        super().__init__(observation_space, action_space)
        self.policy = policy
        self.value = value
        if share_encoder:
            _share_encoder(self.modules())

    def forward(self, obs):
        policy = self.policy(obs)
        value = self.value(obs)

        return policy, value


# Agent

class Agent:
    def __init__(
            self,
            model: TorchModel,
            num_envs: int,
            buffer: Buffer,
    ):
        self.model = model
        self.num_envs = num_envs
        self.buffer = buffer

    def act(self, obs):
        raise NotImplementedError

    def collect(self):
        raise NotImplementedError
