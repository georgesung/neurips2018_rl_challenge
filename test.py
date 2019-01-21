from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import yaml
import argparse
import numpy as np

import ray
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
import ray.rllib.agents.ddpg as ddpg
import ray.rllib.agents.ppo as ppo

from osim.env import ProstheticsEnv
from osim.http.client import Client

from utils import ProstheticsEnvWrapper

# Command line parameters
parser = argparse.ArgumentParser(description='Hello! Make more friendly and flexible later')
parser.add_argument('--token', dest='token', action='store', required=False)
parser.add_argument('--visualize', dest='visualize', action='store_true', default=False)
parser.add_argument('--checkpoint', dest='checkpoint', required=True)
parser.add_argument('--expyaml', dest='expyaml', required=True)
parser.add_argument('--loops', dest='loops', required=True)
args = parser.parse_args()

ray.init()

# Load experiment config
exp_config = yaml.load(open(args.expyaml, 'r'))
exp_name = list(exp_config.keys())[0]
state_norm_file = exp_config[exp_name]['state_norm_file']
downsample_factor = exp_config[exp_name]['downsample_factor']
agent_type = exp_config[exp_name]['run']
#integrator_accuracy = exp_config[exp_name]['integrator_accuracy']
#checkpoint_period = exp_config[exp_name]['checkpoint_period']
#best_agents_dir = exp_config[exp_name]['best_agents_dir']
#load_prev_agent = exp_config[exp_name]['load_prev_agent']
#custom_reward = exp_config[exp_name]['custom_reward']
#custom_reward_params = exp_config[exp_name]['custom_reward_params']

# Load agent config, force 0 workers (i.e. only head "node")
agent_config = exp_config[exp_name]['agent_config']
agent_config['num_workers'] = 0

# Set up environment
def create_env(env_config):
    env = ProstheticsEnv(**env_config)
    env = ProstheticsEnvWrapper(
        env,
        do_norm=True,
        state_norm_file=state_norm_file,
        custom_reward=False,
        downsample_factor=downsample_factor,
        reduce_action=True,
        test_mode=True)
    return env
register_env("prosthetics_env", create_env)

# Set up agent (agent references environment)
# Use default integrator accuracy in test mode
#agent_config['env_config'] = {'visualize': not args.token and args.visualize, 'integrator_accuracy': integrator_accuracy}
agent_config['env_config'] = {'visualize': not args.token and args.visualize, "difficulty": 0}
print('agent_config:\n%s' % (agent_config,))

if agent_type == 'DDPG':
    agent = ddpg.DDPGAgent(
        env="prosthetics_env",
        config=agent_config
    )
elif agent_type == 'PPO':
    agent = ppo.PPOAgent(
        env="prosthetics_env",
        config=agent_config
    )
else:
    raise ValueError('Unspported agent type')

agent.restore(args.checkpoint)

# Different inference procedure if using LSTM or not
use_lstm = False
if 'model' in agent_config:
    if 'use_lstm' in agent_config['model']:
        if agent_config['model']['use_lstm']:
            use_lstm = True

if args.token:
    # Submit to competition
    # Reference: https://github.com/stanfordnmbl/osim-rl/blob/master/examples/submit.py

    remote_base = 'http://grader.crowdai.org:1729' # Submission to Round-1
    #remote_base = 'http://grader.crowdai.org:1730' # Submission to Round-2
    crowdai_token = args.token

    # Dummy environment, just need process_state_desc()
    dummy_env = create_env(agent_config['env_config'])

    # Create environment w/ Client
    client = Client(remote_base)
    state_desc = client.env_create(crowdai_token, env_id="ProstheticsEnv")
    state = dummy_env.process_state_desc(state_desc)  # initial state

    if use_lstm:
        # Initial hidden state at start of episode
        hidden = agent.local_evaluator.policy_map['default'].get_initial_state()

    # Evaluation loop
    while True:
        # NOTE TODO: reduce action space is hard-coded in train.py!
        if use_lstm:
            action, hidden, logits_dict = agent.compute_action(observation=state, state=hidden)
        else:
            action = agent.compute_action(state)

        action = dummy_env.expand_action(action)  # get back original action space, this is also a list now

        # Repeat same action downsample_factor number of times
        for _ in range(downsample_factor):
            state_desc, reward, done, info = client.env_step(action) #, True)
            if done:
                break

        state = dummy_env.process_state_desc(state_desc)  # "next state"

        if done:
            state_desc = client.env_reset()
            if not state_desc:
                break
            state = dummy_env.process_state_desc(state_desc)

            if use_lstm:
                # Initial hidden state at start of episode
                hidden = agent.local_evaluator.policy_map['default'].get_initial_state()

    client.submit()
else:
    # Test and visualize locally
    env = create_env(agent_config['env_config'])
    for ep_num in range(int(args.loops)):
        state = env.reset()
        ep_reward = 0.

        if use_lstm:
            # Initial hidden state at start of episode
            hidden = agent.local_evaluator.policy_map['default'].get_initial_state()

        for i in range(9999):
            if use_lstm:
                action, hidden, logits_dict = agent.compute_action(observation=state, state=hidden)
            else:
                action = agent.compute_action(state)

            state, reward, done, info = env.step(action)
            ep_reward += reward
            if done:
                break

        print('Episodic reward: %f, (approximate) episode length: %d' % (ep_reward, i*downsample_factor))

    _ = env.reset()
