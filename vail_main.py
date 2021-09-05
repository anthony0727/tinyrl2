import pickle
from collections import namedtuple
from copy import deepcopy

import gym
import torch
from marlenv.wrappers import make_snake
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from rl2.buffers import DynamicReplayBuffer

with open('/home/anthony/data/human.pickle', 'rb') as fp:
    raw_data = pickle.load(fp)

field_names = ('obs', 'action', 'reward', 'done')
Transition = namedtuple(
    'Transition',
    field_names=field_names,
)


class Worker:
    def __init__(self, env, agent):
        self.env = env
        self.agent = agent

        # current infos
        self.step = 0
        self.transition = Transition(env.reset(), None, None, False)

        # sanity check
        dummy_output = self.dummy_forward()

        # now initialize

        # for static buffer, determine shape and malloc, may take a while
        agent.buffer.allocate(dummy_output)

    def rollout(self):
        action = self.agent.act(self.transition.obs)
        obs, reward, done, env_info = env.step(action)
        self.transition = Transition(obs, action, reward, done)

    def run(self, num_steps):
        target_step = self.step + num_steps
        for ith_step in range(self.step, target_step):
            self.rollout()

    def dummy_forward(self):
        dummy_env = deepcopy(env)
        obs = dummy_env.reset()
        dummy_output = self.agent.model(self.obs)

        return dummy_output


TF_TO_PT = (0, 3, 1, 2)


class TorchEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obses, rews, dones, infos = self.env.step(action)
        if isinstance(obses, list):
            obses = [obs.transpose(*TF_TO_PT) for obs in obses]
        else:
            obses = obses.transpose(*TF_TO_PT)

        return obses, rews, dones, infos


class ActionOneHot2d(nn.Module):
    def __init__(self, num_classes, data_shape):
        super(ActionOneHot2d, self).__init__()
        assert len(data_shape) == 2, 'must be 2d image shape'
        self.eval()
        self.num_classes = num_classes
        self.data_shape = data_shape
        embeddings = []
        # change this to torch.tile for torch version > 1.6
        for i in range(num_classes):
            embedding = torch.zeros(num_classes, *data_shape)
            embedding[i] = 1.
            embeddings.append(embedding)

        # TODO: shape management
        self.embeddings = torch.stack(embeddings).transpose(1, -1)
        self.shape = self.embeddings.shape

    def __getitem__(self, idx):
        return self.embeddings[idx]

    def forward(self, x):
        return self.embeddings[x]


def reparameterize(mu, sigma):
    eps = torch.randn_like(sigma)
    return mu + eps * sigma


def loss_fn(obs):
    return (actor(obs) + critic(obs)).mean()


def train_discriminator():
    optimizer = Adam(discriminator.parameters())

    num_epochs = 1
    for epoch in range(num_epochs):
        for batch in data:
            mu, sigma = get_mu(batch), get_sigma(batch)
            z = reparameterize(mu, sigma)
            logit = discriminator(z)

            loss = nn.BCEWithLogitsLoss(logit, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    mu, sigma = get_mu(batch), get_sigma(batch)
    z = reparameterize(mu, sigma)
    logit = discriminator(z)

    buffer.reward = nn.LogSigmoid(logit)


def train_rl():
    params = list(actor.parameters()) + list(critic[1:].parameters())
    optimizer = Adam(params)

    num_steps = int(1e6)
    for step in (num_steps):
        num_epochs = 1
        for epoch in range(num_epochs):
            for batch in data_loader:
                action_dist = actor(batch)
                value_dist = critic(batch)
                # TODO: calc loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()


class ActorCritic(nn.Module):
    def __init__(self, actor, critic):
        self.actor = actor
        self.critic = critic

    def forward(self, obs):
        # forward until just before sth we need for both act and train
        action_dist = self.actor(obs)
        value_dist = self.critic(obs)

        return action_dist, value_dist


class PPOAgent:
    def __init__(self, model, buffer):
        self.model = model
        self.buffer = buffer

    def act(self, obs):
        action_dist = self.model(obs)
        action = action_dist.sample().squeeze()

        return action


if __name__ == '__main__':
    # most outer client code
    buffer = DynamicReplayBuffer()
    data = DataLoader(buffer.data)  # can we do DataLoader(buffer) instead?
    num_envs = 64

    env, state_dim, action_dim, props = make_snake(num_envs=num_envs, num_snakes=1, frame_stack=2, vision_range=5)
    env = TorchEnv(env)

    width, height, channels = state_dim
    input_shape = (num_envs, channels, height, width)
    obs = env.reset().reshape(*input_shape)

    # state_dim
    width, height, channels = state_dim

    encoder = nn.Sequential(
        nn.Conv2d(channels, 64, 3),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3),
        nn.Flatten()
    )

    with torch.no_grad():
        dummy = torch.Tensor(obs)
        encoded_dim = encoder(dummy).shape[-1]
    # encoded_dim

    latent_dim = 128
    get_mu = nn.Sequential(
        encoder,
        nn.Linear(encoded_dim, 64),
        nn.Tanh(),
        nn.Linear(64, latent_dim),
    )

    get_sigma = nn.Sequential(
        encoder,
        nn.Linear(encoded_dim, 64),
        nn.Tanh(),
        nn.Linear(64, latent_dim),
    )

    discriminator = nn.Sequential(
        nn.Linear(latent_dim, 64),
        nn.Tanh(),
        nn.Linear(64, 1)
    )

    continuous = False
    if continuous:
        squash_fn = nn.Tanh()
    else:
        squash_fn = nn.Softmax(dim=-1)

    actor = nn.Sequential(
        encoder,
        nn.Linear(encoded_dim, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, action_dim[0]),
        squash_fn,
    )

    critic = nn.Sequential(
        encoder,
        nn.Linear(encoded_dim, 64),
        nn.Tanh(),
        nn.Linear(64, 64),
        nn.Tanh(),
        nn.Linear(64, action_dim[0]),
        nn.Softmax(dim=-1)
    )

    framework = ActorCritic(actor=actor, critic=critic)
    agent = PPOAgent(
        framework=framework,
        buffer=buffer,
        loss_fn=loss_fn
    )

    worker = Worker(
        env=env,
        agent=agent
    )
    worker.run(100)
    # additional training
    worker.run(100)
