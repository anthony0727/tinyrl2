import os
import random
from pathlib import Path

import numpy as np

WORKING_DIR = os.path.join(Path.home(), 'rl2-runs')
CACHE_DIR = os.path.join(WORKING_DIR, '.cache')


def global_seed(seed=42):
    from tinyrl2.ctx import config_ctx
    random.seed(seed)
    np.random.seed(seed)
    try:
        from torch import random as pt_random
        pt_random.seed(seed)
    except:
        pass

    try:
        from tensorflow import random as tf_random
        tf_random.set_seed(seed)
    except:
        pass

    config_ctx.set({'seed': seed})

    return seed
