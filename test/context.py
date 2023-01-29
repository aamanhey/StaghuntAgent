import os
import sys

# Add src code to test dir scope
path = os.getcwd()
parent_dir = os.path.abspath(os.path.join(path, os.pardir))
src_dir = os.path.abspath(os.path.join(parent_dir, 'src'))
sys.path.insert(0, src_dir)

# import <module>
# Add to each test file: from .context import <module>
from state import State
from setup import map_init
from encoder import Encoder, StaghuntEncoder
from feature_extractor import StaghuntExtractor
from game_configs import simple_map, character_setup_1h1s1r, custom_map,
from interaction_manager import InteractionManager, RABBIT_VALUE, STAG_VALUE
