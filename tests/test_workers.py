import gym

from tinyrl2.agents import RandomAgent
from tinyrl2.models.random import RandomModel
from tinyrl2.workers import StepWorker, EpisodeWorker


def test_step_worker():
    env = gym.make('CartPole-v1')
    agent = RandomAgent(
        RandomModel(
            env.observation_space,
            env.action_space
        )
    )
    step_worker = StepWorker(env, agent)
    step_worker.run(num_steps=10)


def test_episode_worker():
    env = gym.make('CartPole-v1')
    agent = RandomAgent(
        RandomModel(
            env.observation_space,
            env.action_space
        )
    )
    episode_worker = EpisodeWorker(env, agent)
    episode_worker.run(num_episodes=5)
