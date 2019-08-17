# -*- coding: utf-8 -*-

import os
import logging
import logging.config

PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../..'))
RESOURCE_PATH = os.path.join(PROJECT_PATH, 'resource')
DATA_PATH = os.path.join(PROJECT_PATH, 'data')
PRETRAINED_MODEL_PATH = os.path.join(RESOURCE_PATH, 'model')
SAVED_MODEL_PATH = os.path.join(DATA_PATH, 'model')


# logging.
logging.config.fileConfig(os.path.join(RESOURCE_PATH, 'config', 'logger.conf'))
logger = logging.getLogger('root')
