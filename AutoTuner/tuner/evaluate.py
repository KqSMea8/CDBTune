# -*- coding: utf-8 -*-
"""
description: Evaluate the Model
"""

import os
import sys
import utils
import pickle
import argparse
import tuner_configs
sys.path.append('../')
import models
import environment


parser = argparse.ArgumentParser()
parser.add_argument('--tencent', action='store_true', help='Use Tencent Server')
parser.add_argument('--params', type=str, required=True, help='Load existing parameters')
parser.add_argument('--workload', type=str, default='read', help='Workload type [`read`, `write`, `readwrite`]')
parser.add_argument('--instance', type=str, default='mysql1', help='Choose MySQL Instance')
parser.add_argument('--method', type=str, default='ddpg', help='Choose Algorithm to solve [`ddpg`,`dqn`]')
parser.add_argument('--memory', type=str, default='', help='add replay memory')

opt = parser.parse_args()

# Create Environment
if opt.tencent:
    env = environment.TencentServer(wk_type=opt.workload, instance_name=opt.instance, request_url=tuner_configs.TENCENT_URL)
else:
    env = environment.Server(wk_type=opt.workload, instance_name=opt.instance)

tconfig = tuner_configs.config

# Build models
if opt.method == 'ddpg':

    ddpg_opt = dict()
    ddpg_opt['tau'] = 0.001
    ddpg_opt['alr'] = 0.0001
    ddpg_opt['clr'] = 0.0001
    ddpg_opt['model'] = opt.params
    ddpg_opt['gamma'] = tconfig['gamma']
    ddpg_opt['batch_size'] = tconfig['batch_size']
    ddpg_opt['memory_size'] = tconfig['memory_size']

    model = models.DDPG(
        n_states=tconfig['num_states'],
        n_actions=tconfig['num_actions'],
        opt=ddpg_opt,
        mean_var_path='mean_var.pkl', # or None
        ouprocess=True
    )

else:

    model = models.DQN()
    pass

if not os.path.exists('log'):
    os.mkdir('log')


expr_name = 'eval_{}_{}'.format(opt.method, str(utils.get_timestamp()))

logger = utils.Logger(
    name=opt.method,
    log_file='log/{}.log'.format(expr_name)
)

# Load mean value and variance
with open('mean_var.pkl', 'rb') as f:
    mean, var = pickle.load(f)

current_knob = environment.get_init_knobs()


def generate_knob(action, method):
    if method == 'ddpg':
        return environment.gen_continuous(action)
    else:
        raise NotImplementedError()


if len(opt.memory) > 0:
    model.replay_memory.load_memory(opt.memory)
    print("Load Memory: {}".format(len(model.replay_memory)))

step_counter = 0
train_step = 0
if opt.method == 'ddpg':
    accumulate_loss = [0, 0]
else:
    accumulate_loss = 0

max_step = 0
max_value = 0.0
generate_knobs = []
current_state = env.initialize()
model.reset(0.1)
while step_counter < 20:
    state = current_state
    action = model.choose_action(state)
    if opt.method == 'ddpg':
        current_knob = generate_knob(action, 'ddpg')
        logger.info("[ddpg] Action: {}".format(action))
    else:
        action, qvalue = action
        current_knob = generate_knob(action, 'dqn')
        logger.info("[dqn] Q:{} Action: {}".format(qvalue, action))

    reward, state_, done, score, metrics = env.step(current_knob)
    logger.info("[{}][Step: {}][Metric tps:{} lat:{}, qps: {}]Reward: {} Score: {} Done: {}".format(
        opt.method, step_counter, metrics[0], metrics[1], metrics[2], reward, score, done
    ))

    next_state = state_

    model.replay_memory.push(
        state=state,
        reward=reward,
        action=action,
        next_state=next_state,
        terminate=done
    )

    current_state = next_state
    step_counter += 1
    generate_knobs.append((score, current_knob))
    if max_value < score:
        max_step = step_counter-1
        max_value = score
    if len(model.replay_memory) >= tconfig['batch_size']:
        losses = []
        for i in xrange(2):
            losses.append(model.update())
            train_step += 1

        if opt.method == 'ddpg':
            accumulate_loss[0] += sum([x[0] for x in losses])
            accumulate_loss[1] += sum([x[1] for x in losses])
            logger.info('[{}][Step: {}] Critic: {} Actor: {}'.format(
                opt.method, step_counter, accumulate_loss[0] / train_step, accumulate_loss[1] / train_step
            ))
        else:
            accumulate_loss += sum(losses)
            logger.info('[{}][Step: {}] Loss: {}'.format(
                opt.method, step_counter, accumulate_loss / train_step
            ))

    if done:
        current_state = env.initialize()
        model.reset(0.01)


print("Searching Finished")
with open(expr_name + '.pkl', 'wb') as f:
    pickle.dump(generate_knobs, f)

print("Knobs are saved!")
# eval

default_konbs = environment.get_init_knobs()
max_knobs = generate_knobs[max_step][1]

metric1 = env.eval(default_konbs)
print("Default TPS: {} Latency: {}".format(metric1['tps'], metric1['latency']))
metric2 = env.eval(max_knobs)
print("Max TPS: {} Latency: {}".format(metric2['tps'], metric2['latency']))

delta_tps = (metric2['tps'] - metric1['tps']) / metric1['tps']
delta_latency = (-metric2['latency'] + metric1['latency']) / metric1['latency']

print("[Evaluation Result] Latency Decrease: {} TPS Increase: {}".format(delta_latency, delta_tps))
