import logging

import numpy as np

import gym

from tinyrl2.agents import Agent

logger = logging.getLogger(__name__)


class Statistic:
    def __init__(
            self,
            score,
            steps,
    ):
        self.score = score
        self.steps = steps

    def __repr__(self):
        return f'score: {self.score} steps: {self.steps}'


class RolloutWorker:
    """
    relations - agent to env : 1 to N
    """

    def __init__(
            self,
            env: gym.Env,
            agent: Agent,
    ):
        self.env = env
        self.agent = agent

        self.obs, self.done = self._resolve_env()

        self.curr_step = 0
        self.curr_episode = 0

    def _resolve_env(self):
        obs = self.env.reset()
        done = [False] * self.agent.num_envs

        return obs, done

    def rollout(self):
        action = self.agent.act(self.obs)
        self.obs, rew, self.done, info = self.env.step(action)

    def _postprocess_episode(self, num_steps):
        self.curr_episode += 1

        score = 0
        stat = Statistic(score=score, steps=num_steps)
        return stat

    def add_statistic(self, aa):
        pass

    def all_done(self):
        return np.all(self.done)

    def run(self):
        raise NotImplementedError


class StepWorker(RolloutWorker):
    def __init__(
            self,
            env: gym.Env,
            agent: Agent,
    ):
        super().__init__(env, agent)

    def run(self, num_steps):
        _num_steps = 0
        target_step = self.curr_step + num_steps
        for step in range(self.curr_step, target_step):
            _num_steps += self.agent.num_envs
            self.rollout()

            if self.all_done():
                episode_stat = self._postprocess_episode(_num_steps)
                self.add_statistic(episode_stat)


class EpisodeWorker(RolloutWorker):
    def __init__(
            self,
            env: gym.Env,
            agent: Agent,
    ):
        super().__init__(env, agent)

    def run(self, num_episodes):
        _num_steps = 0
        target_episode = self.curr_episode + num_episodes
        while self.curr_episode < target_episode:
            _num_steps += self.agent.num_envs
            self.rollout()

            if self.all_done():
                episode_stat = self._postprocess_episode(_num_steps)
                self.add_statistic(episode_stat)
