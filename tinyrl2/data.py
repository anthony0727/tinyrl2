import logging
import os
from pathlib import Path

import wget as wget

from tinyrl2 import CACHE_DIR

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)


def get_snake_expert_trajectories(key='ppo'):
    url = "https://rl2.s3.ap-northeast-2.amazonaws.com/imitation-expert-trajectory"

    cache_path = Path(CACHE_DIR)
    cache_path.mkdir(parents=True, exist_ok=True)
    # filenames = list(cache_path.iterdir())
    filenames = os.listdir(cache_path.absolute())

    if key == 'ppo':
        filename = 'ppo_10.pickle'
    elif key == 'human':
        filename = 'human.pickle'

    logger.info(f'downloading expert trajectory... {filename}')
    if filename in filenames:
        logger.info(f'file already exists, using {filename} in {CACHE_DIR}')
    else:
        filename = wget.download(f'{url}/{filename}', CACHE_DIR)
    logger.info(f'saved to {filename}')

    return filename