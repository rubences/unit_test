"""
Bio-Adaptive Haptic Coaching: Proof-of-Concept

Main module for the 4-phase PoC pipeline.
"""

__version__ = '1.0.0'
__author__ = 'Bio-Adaptive Coaching Research Team'

from . import data_gen
from . import env
from . import train
from . import vis

__all__ = ['data_gen', 'env', 'train', 'vis']
