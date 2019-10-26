# -*- coding: utf-8 -*-
"""
description: Tuner's configurations
"""

config = {
    'num_actions': 16,
    'num_states': 63,
    'use_gpu': False,
    'epsilon': 0.1, #自行修改
    'gamma': 0.09, #自行修改
    'update_target': 20,
    'epoches': 100,
    'batch_size': 16,
    'learning_rate': 0.01, #自行修改
    'memory_size': 100000,
}
