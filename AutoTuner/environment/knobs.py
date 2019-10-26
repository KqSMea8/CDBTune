# -*- coding: utf-8 -*-
"""
desciption: Knob information

"""

import utils
import configs


memory_size = 32*1024*1024*1024
instance_name = ''

'''
在实验中选择可调的KNOBS个数为16，默认值由腾讯公司DBA的经验给出，这里不方便显示
'''
KNOBS = ['skip_name_resolve',               #
         'table_open_cache',                #
         'max_connections',                 #
         'innodb_buffer_pool_size',         #
         'innodb_buffer_pool_instances',    #
         'innodb_log_files_in_group',       #
         'innodb_log_file_size',            #
         'innodb_purge_threads',            #
         'innodb_read_io_threads',          #
         'innodb_write_io_threads',         #
         'innodb_file_per_table',           #
         'binlog_checksum',                 #
         'binlog_cache_size',               #
         'max_binlog_cache_size',           #
         'max_binlog_size',                 #
         'binlog_format'                    #
         ]

KNOB_DETAILS = None
num_knobs = len(KNOBS)


def init_knobs(instance):
    global instance_name
    global memory_size
    global KNOB_DETAILS
    instance_name = instance
    memory_size = configs.instance_config[instance]['memory']

    DEFAULT_VALUE = 0 # 这个值为knobs的默认值，按理来说各个knobs的默认值不一样，可以根据各自的经验进行修改
    KNOB_DETAILS = {
        'skip_name_resolve': ['enum', ['ON', 'OFF']],
        'table_open_cache': ['integer', [1, 524288, DEFAULT_VALUE]],
        'max_connections': ['integer', [1100, 100000, DEFAULT_VALUE]],
        'innodb_buffer_pool_size': ['integer', [1048576, memory_size, DEFAULT_VALUE]],
        'innodb_buffer_pool_instances': ['integer', [1, 64, DEFAULT_VALUE]],
        'innodb_log_files_in_group': ['integer', [2, 100, DEFAULT_VALUE]],
        'innodb_log_file_size': ['integer', [1048576, 5497558138, DEFAULT_VALUE]],
        'innodb_purge_threads': ['integer', [1, 32, DEFAULT_VALUE]],
        'innodb_read_io_threads': ['integer', [1, 64, DEFAULT_VALUE]],
        'innodb_write_io_threads': ['integer', [1, 64, DEFAULT_VALUE]],
        'innodb_file_per_table': ['enum', ['OFF', 'ON']],
        'binlog_checksum': ['enum', ['NONE', 'CRC32']],
        'binlog_cache_size': ['integer', [1048, 34359738368, DEFAULT_VALUE]],
        'max_binlog_cache_size': ['integer', [4096, 4294967296, DEFAULT_VALUE]],
        'max_binlog_size': ['integer', [4096, 1073741824, DEFAULT_VALUE]],
        'binlog_format': ['enum', ['ROW', 'MIXED']],
    }

    print("Instance: %s Memory: %s" % (instance_name, memory_size))


def get_init_knobs():

    knobs = {}

    for name, value in KNOB_DETAILS.items():
        knob_value = value[1]
        knobs[name] = knob_value[-1]

    return knobs


def gen_continuous(action):
    knobs = {}

    for idx in xrange(num_knobs):
        name = KNOBS[idx]
        value = KNOB_DETAILS[name]
        knob_type = value[0]
        knob_value = value[1]
        min_value = knob_value[0]
        if knob_type == 'integer':
            max_val = knob_value[1]
            eval_value = int(max_val * action[idx])
            eval_value = max(eval_value, min_value)
        else:
            enum_size = len(knob_value)
            enum_index = int(enum_size * action[idx])
            enum_index = min(enum_size - 1, enum_index)
            eval_value = knob_value[enum_index]

        if name == 'innodb_log_file_size':
            max_val = 32 * 1024 * 1024 * 1024 / knobs['innodb_log_files_in_group']
            eval_value = int(max_val * action[idx])
            eval_value = max(eval_value, min_value)
        knobs[name] = eval_value

    return knobs


def save_knobs(knob, metrics, knob_file):
    """ Save Knobs and their metrics to files
    Args:
        knob: dict, knob content
        metrics: list, tps and latency
        knob_file: str, file path
    """
    # format: tps, latency, knobstr: [#knobname=value#]
    knob_strs = []
    for kv in knob.items():
        knob_strs.append('{}:{}'.format(kv[0], kv[1]))
    result_str = '{},{},{},'.format(metrics[0], metrics[1], metrics[2])
    knob_str = "#".join(knob_strs)
    result_str += knob_str

    with open(knob_file, 'a+') as f:
        f.write(result_str+'\n')


