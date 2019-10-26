# -*- coding: utf-8 -*-
"""
Train the model
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
parser.add_argument('--params', type=str, default='', help='Load existing parameters')
parser.add_argument('--workload', type=str, default='read', help='Workload type [`read`, `write`, `readwrite`]')
parser.add_argument('--instance', type=str, default='mysql1', help='Choose MySQL Instance')
parser.add_argument('--method', type=str, default='ddpg', help='Choose Algorithm to solve [`ddpg`,`dqn`]')
parser.add_argument('--memory', type=str, default='', help='add replay memory')
parser.add_argument('--noisy', action='store_true', help='use noisy linear layer')

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
    ddpg_opt['tau'] = 0.002
    ddpg_opt['alr'] = 0.0005
    ddpg_opt['clr'] = 0.0001
    ddpg_opt['model'] = opt.params
    ddpg_opt['gamma'] = tconfig['gamma']
    ddpg_opt['batch_size'] = tconfig['batch_size']
    ddpg_opt['memory_size'] = tconfig['memory_size']

    model = models.DDPG(
        n_states=tconfig['num_states'],
        n_actions=tconfig['num_actions'],
        opt=ddpg_opt,
      #  mean_var_path='mean_var.pkl',
        ouprocess=not opt.noisy
    )

else:

    model = models.DQN()
    pass

if not os.path.exists('log'):
    os.mkdir('log')

if not os.path.exists('save_memory'):
    os.mkdir('save_memory')

if not os.path.exists('save_knobs'):
    os.mkdir('save_knobs')

if not os.path.exists('save_state_actions'):
    os.mkdir('save_state_actions')

if not os.path.exists('model_params'):
    os.mkdir('model_params')

expr_name = 'train_{}_{}'.format(opt.method, str(utils.get_timestamp()))

logger = utils.Logger(
    name=opt.method,
    log_file='log/{}.log'.format(expr_name)
)

# # Load mean value and variance
# with open('mean_var.pkl', 'rb') as f:
#     mean, var = pickle.load(f)

current_knob = environment.get_init_knobs()


def generate_knob(action, method):
    if method == 'ddpg':
        return environment.gen_continuous(action)
    else:
        raise NotImplementedError('Not Implemented')


# OUProcess
origin_sigma = 0.10
sigma = origin_sigma
# decay rate
sigma_decay_rate = 0.99
step_counter = 0
train_step = 0
if opt.method == 'ddpg':
    accumulate_loss = [0, 0]
else:
    accumulate_loss = 0

fine_state_actions = []

if len(opt.memory) > 0:
    model.replay_memory.load_memory(opt.memory)
    print("Load Memory: {}".format(len(model.replay_memory)))

for episode in xrange(tconfig['epoches']):
    current_state, initial_metrics = env.initialize()
    logger.info("\n[Env initialized][Metric tps: {} lat: {} qps: {}]".format(
        initial_metrics[0], initial_metrics[1], initial_metrics[2]))

    model.reset(sigma)
    t = 0
    while True:
        state = current_state
        if opt.noisy:
            model.sample_noise()
        action = model.choose_action(state)
        if opt.method == 'ddpg':
            current_knob = generate_knob(action, 'ddpg')
            logger.info("[ddpg] Action: {}".format(action))
        else:
            action, qvalue = action
            current_knob = generate_knob(action, 'dqn')
            logger.info("[dqn] Q:{} Action: {}".format(qvalue, action))

        reward, state_, done, score, metrics = env.step(current_knob)
        logger.info("\n[{}][Episode: {}][Step: {}][Metric tps:{} lat:{} qps:{}]Reward: {} Score: {} Done: {}".format(
            opt.method, episode, t, metrics[0], metrics[1], metrics[2], reward, score, done
        ))

        next_state = state_

        model.replay_memory.push(
            state=state,
            reward=reward,
            action=action,
            next_state=next_state,
            terminate=done
        )

        if score > 5:
            fine_state_actions.append((state, action))

        current_state = next_state
        t = t + 1
        step_counter += 1

        if len(model.replay_memory) > tconfig['batch_size']:
            losses = []
            for i in xrange(2):
                losses.append(model.update())
                train_step += 1

            if opt.method == 'ddpg':
                accumulate_loss[0] += sum([x[0] for x in losses])
                accumulate_loss[1] += sum([x[1] for x in losses])
                logger.info('[{}][Episode: {}][Step: {}] Critic: {} Actor: {}'.format(
                    opt.method, episode, t-1, accumulate_loss[0]/train_step, accumulate_loss[1]/train_step
                ))
            else:
                accumulate_loss += sum(losses)
                logger.info('[{}][Episode: {}][Step: {}] Loss: {}'.format(
                    opt.method, episode, t-1, accumulate_loss/train_step
                ))

        # save replay memory
        if step_counter % 10 == 0:
            model.replay_memory.save('save_memory/{}.pkl'.format(expr_name))
            utils.save_state_actions(fine_state_actions, 'save_state_actions/{}.pkl'.format(expr_name))
            # sigma = origin_sigma*(sigma_decay_rate ** (step_counter/10))

        # save network
        if step_counter % 50 == 0:
            model.save_model('model_params', title='{}_{}'.format(expr_name, step_counter))

        if done or score < -50:
            break



