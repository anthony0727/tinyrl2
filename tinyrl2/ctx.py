import os
from pathlib import Path
from contextvars import ContextVar

from tinyrl2 import WORKING_DIR

default_config = {
    'working_dir': WORKING_DIR
}
config_ctx = ContextVar('config', default=default_config)
