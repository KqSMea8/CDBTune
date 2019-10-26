# -*- coding: utf-8 -*-
"""
description: MySQL Environment
"""

import re
import os
import time
import datetime
import json
import threading
import MySQLdb
import numpy as np
import configs
import utils
import knobs
import requests


# TEMP_FILES = "/data/AutoTuner/train_result/tmp/"
# PROJECT_DIR = "/data/"
TEMP_FILES = "/home/rmw/train_result/tmp/"
PROJECT_DIR = "/home/rmw/"


class MySQLEnv(object):

    def __init__(self, wk_type='read', alpha=1.0, beta1=0.5, beta2=0.5, time_decay1=1.0, time_decay2=1.0):

        self.db_info = None
        self.wk_type = wk_type
        self.score = 0.0
        self.steps = 0
        self.terminate = False
        self.last_external_metrics = None
        self.default_externam_metrics = None

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.time_decay_1 = time_decay1
        self.time_decay_2 = time_decay2

    @staticmethod
    def _get_external_metrics(path):

        def parse_sysbench_new(file_path):
            with open(file_path) as f:
                lines = f.read()
            temporal_pattern = re.compile(
                "tps: (\d+.\d+) qps: (\d+.\d+) \(r/w/o: (\d+.\d+)/(\d+.\d+)/(\d+.\d+)\)" 
                " lat \(ms,95%\): (\d+.\d+) err/s: (\d+.\d+) reconn/s: (\d+.\d+)")
            temporal = temporal_pattern.findall(lines)
            tps = 0
            latency = 0
            qps = 0

            for i in temporal[5:]:
                tps += float(i[0])
                latency += float(i[5])
                qps += float(i[1])
            num_samples = len(temporal[5:])
            tps /= num_samples
            qps /= num_samples
            latency /= num_samples
            return [tps, latency, qps]

        result = parse_sysbench_new(path)

        return result

    def _get_internal_metrics(self, internal_metrics):
        """
        Args:
            internal_metrics: list,
        Return:

        """
        _counter = 0
        _period = 5
        count = 12

        def collect_metric(counter):
            counter += 1
            timer = threading.Timer(_period, collect_metric, (counter,))
            timer.start()
            if counter >= count:
                timer.cancel()
            try:
                data = utils.get_metrics(self.db_info)
                internal_metrics.append(data)
            except MySQLdb.Error as e:
                print("[GET Metrics]Exception:%s" % e.message)

        collect_metric(_counter)

        return internal_metrics

    @staticmethod
    def _post_handle(metrics):
        result = np.zeros(63)

        def do(metric_name, metric_values):
            metric_type = utils.get_metric_type(metric_name)
            if metric_type == 'counter':
                return float(metric_values[-1] - metric_values[0])
            else:
                return float(sum(metric_values))/len(metric_values)

        keys = metrics[0].keys()
        keys.sort()
        for idx in xrange(len(keys)):
            key = keys[idx]
            data = [x[key] for x in metrics]
            result[idx] = do(key, data)

        return result

    def initialize(self):
        """Initialize the mysql instance environment
        """
        pass

    def eval(self, knob):
        """ Evaluate the knobs
        Args:
            knob: dict, mysql parameters
        Returns:
            result: {tps, latency}
        """
        flag = self._apply_knobs(knob)
        if not flag:
            return {"tps": 0, "latency": 0}

        external_metrics, _ = self._get_state()
        return {"tps": external_metrics[0],
                "latency": external_metrics[1]}

    def step(self, knob):
        """step
        """
        flag = self._apply_knobs(knob)
        if not flag:
            return -100.0, np.array([0] * 63), True, self.score - 100, [0, 0, 0]

        external_metrics, internal_metrics = self._get_state()
        reward = self._get_reward(external_metrics)
        self.last_external_metrics = external_metrics
        next_state = internal_metrics
        terminate = self._terminate()
        knobs.save_knobs(
            knob=knob,
            metrics=external_metrics,
            knob_file='%sAutoTuner/tuner/save_knobs/knob_metric.txt' % PROJECT_DIR
        )
        return reward, next_state, terminate, self.score, external_metrics

    def _get_state(self):
        """Collect the Internal State and External State
        """
        filename = TEMP_FILES
        if not os.path.exists(filename):
            os.mkdir(filename)
        timestamp = int(time.time())
        filename += '%s.txt' % timestamp
        internal_metrics = []
        self._get_internal_metrics(internal_metrics)

        os.system("bash %sAutoTuner/scripts/run_sysbench.sh %s %s %d %s %s" % (PROJECT_DIR,
                                                                               self.wk_type,
                                                                               self.db_info['host'],
                                                                               self.db_info['port'],
                                                                               self.db_info['passwd'],
                                                                               filename))

        time.sleep(10)

        external_metrics = self._get_external_metrics(filename)
        internal_metrics = self._post_handle(internal_metrics)

        return external_metrics, internal_metrics

    def _apply_knobs(self, knob):
        """ Apply Knobs to the instance
        """
        pass

    @staticmethod
    def _calculate_reward(delta0, deltat):

        if delta0 > 0:
            _reward = ((1+delta0)**2-1) * (1+deltat)
        else:
            _reward = - ((1-delta0)**2-1) * (1-deltat)
        if _reward and deltat < 0:
            _reward = 0
        return _reward

    def _get_reward(self, external_metrics):
        """
        Args:
            external_metrics: list, external metric info, including `tps` and `qps`
        Return:
            reward: float, a scalar reward
        """

        # tps
        delta_0_tps = float((external_metrics[0] - self.default_externam_metrics[0]))/self.default_externam_metrics[0]
        delta_t_tps = float((external_metrics[0] - self.last_external_metrics[0]))/self.last_external_metrics[0]

        tps_reward = self._calculate_reward(delta_0_tps, delta_t_tps)

        # latency
        delta_0_lat = float((-external_metrics[1] + self.default_externam_metrics[1])) / self.default_externam_metrics[1]
        delta_t_lat = float((-external_metrics[1] + self.last_external_metrics[1])) / self.last_external_metrics[1]

        lat_reward = self._calculate_reward(delta_0_lat, delta_t_lat)

        reward = tps_reward * 0.4 + 0.6 * lat_reward
        self.score += reward
        return reward

    def _terminate(self):
        return self.terminate


class Server(MySQLEnv):
    """ Build an environment directly on Server
    """

    def __init__(self, wk_type, instance_name):
        MySQLEnv.__init__(self, wk_type)
        self.wk_type = wk_type
        self.score = 0.0
        self.steps = 0
        self.terminate = False
        self.last_external_metrics = None
        self.instance_name = instance_name
        self.db_info = configs.instance_config[instance_name]
        self.server_ip = self.db_info['host']
        self.alpha = 1.0
        knobs.init_knobs(instance_name)
        self.default_knobs = knobs.get_init_knobs()

    def initialize(self):
        """ Initialize the environment when an episode starts
        Returns:
            state: np.array, current state
        """
        self.score = 0.0
        self.last_external_metrics = []
        self.steps = 0
        self.terminate = False

        flag = self._apply_knobs(self.default_knobs)
        i = 3
        while i >= 0 and not flag:
            flag = self._apply_knobs(self.default_knobs)
            i -= 1
        if i < 0 and not flag:
            print("[Env initializing failed]")
            exit(-1)

        external_metrics, internal_metrics = self._get_state()
        self.last_external_metrics = external_metrics
        self.default_externam_metrics = external_metrics
        state = internal_metrics
        knobs.save_knobs(
            self.default_knobs,
            metrics=external_metrics,
            knob_file='%sAutoTuner/tuner/save_knobs/knob_metric.txt' % PROJECT_DIR
        )
        return state, external_metrics

    def _apply_knobs(self, knob):
        """ Apply the knobs to the mysql
        Args:
            knob: dict, mysql parameters
        Returns:
            flag: whether the setup is valid
        """
        self.steps += 1
        utils.modify_configurations(
            server_ip=self.server_ip,
            instance_name=self.instance_name,
            configuration=knob
        )

        steps = 0
        max_steps = 300
        flag = utils.test_mysql(self.instance_name)
        while not flag and steps < max_steps:
            _st = utils.get_mysql_state(self.server_ip)
#            if not _st:
#                return False
            time.sleep(5)
            flag = utils.test_mysql(self.instance_name)
            steps += 1

        if not flag:
            utils.modify_configurations(
                server_ip=self.server_ip,
                instance_name=self.instance_name,
                configuration=self.default_knobs
            )
            params = ''
            for key in knob.keys():
                params += ' --%s=%s' % (key, knob[key])
            with open('failed.log', 'a+') as f:
                f.write('{}\n'.format(params))
            return False
        else:
            return True


DockerServer = Server

'''
以下为在腾讯公司提供云环境下进行的实验，由于涉及腾讯公司的数据隐私，所以不方便显示
'''
class TencentServer(MySQLEnv):
    pass