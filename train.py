import yaml
import os
import argparse

import ray
from ray.tune.logger import pretty_print
from ray.tune.registry import register_env
import ray.rllib.agents.ddpg as ddpg
import ray.rllib.agents.ppo as ppo

from osim.env import ProstheticsEnv
from utils import ProstheticsEnvWrapper
from gstorage import download_blob, upload_blob, find_latest_checkpoint

# Load experiment config
exp_config = yaml.load(open('train.yaml', 'r'))
exp_name = list(exp_config.keys())[0]
state_norm_file = exp_config[exp_name]['state_norm_file']
downsample_factor = exp_config[exp_name]['downsample_factor']
integrator_accuracy = exp_config[exp_name]['integrator_accuracy']
checkpoint_period = exp_config[exp_name]['checkpoint_period']
output_dir = exp_config[exp_name]['output_dir']
load_prev_agent = exp_config[exp_name]['load_prev_agent']
custom_reward = exp_config[exp_name]['custom_reward']
custom_reward_params = exp_config[exp_name]['custom_reward_params']
agent_type = exp_config[exp_name]['run']

# Load agent config
agent_config = exp_config[exp_name]['agent_config']

# Determine local or gcloud (Google Cloud) run
gcloud_run = type(output_dir) == dict

# Create local output dir
if not os.path.isdir('output_dir'):
    os.system('mkdir output_dir')

# Ray init boilerplate
#ray.init(redis_address='localhost:6379')
ray.init()

# Set up environment
def create_env(env_config):
    env = ProstheticsEnv(**env_config)
    env = ProstheticsEnvWrapper(
        env,
        do_norm=True,
        state_norm_file=state_norm_file,
        custom_reward=custom_reward,
        custom_reward_params=custom_reward_params,
        downsample_factor=downsample_factor,
        reduce_action=True)
    return env
register_env("prosthetics_env", create_env)

# Set up agent (agent references environment)
agent_config['env_config'] = {"visualize": False, "integrator_accuracy": integrator_accuracy, "difficulty": 0}
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

# Restore agent from previous checkpoint file if applicable
if type(load_prev_agent) == str:
    if load_prev_agent == 'latest':
        print('Restoring latest agent from output_dir')

        if gcloud_run:
            # Load latest agent from gcloud storage
            # Find latest checkpoint file in output_dir
            # If no checkpoint exists, then find_latest_checkpoint() will return (None, None)
            extra_data_file, tune_metadata_file = find_latest_checkpoint(
                bucket_name=output_dir['bucket'],
                checkpoint_dir=output_dir['path'])

            # Only restore previous agent if there was a previous checkpoint
            if extra_data_file is not None:
                download_blob(
                    bucket_name=output_dir['bucket'],
                    source_blob_name=extra_data_file,
                    destination_file_name='prev_agent.extra_data')
                download_blob(
                    bucket_name=output_dir['bucket'],
                    source_blob_name=tune_metadata_file,
                    destination_file_name='prev_agent.tune_metadata')
                agent.restore('prev_agent')

                # Also restore logs
                download_blob(
                    bucket_name=output_dir['bucket'],
                    source_blob_name=os.path.join(output_dir['path'], 'best_agents.log'),
                    destination_file_name='output_dir/best_agents.log')
                download_blob(
                    bucket_name=output_dir['bucket'],
                    source_blob_name=os.path.join(output_dir['path'], 'train.log'),
                    destination_file_name='output_dir/train.log')
                download_blob(
                    bucket_name=output_dir['bucket'],
                    source_blob_name=os.path.join(output_dir['path'], 'train_verbose.log'),
                    destination_file_name='output_dir/train_verbose.log')
        else:
            raise ValueError('Load latest agent not supported when running on local machine')
    else:
        # Load previous agent from local file system
        agent.restore(load_prev_agent)

elif type(load_prev_agent) == dict:
    # Continue training from pre-trained agent, that is not stored in output_dir
    print('Restoring agent from %s' % load_prev_agent)
    download_blob(
        bucket_name=load_prev_agent['bucket'],
        source_blob_name=load_prev_agent['path'] + '.extra_data',
        destination_file_name='prev_agent.extra_data')
    download_blob(
        bucket_name=load_prev_agent['bucket'],
        source_blob_name=load_prev_agent['path'] + '.tune_metadata',
        destination_file_name='prev_agent.tune_metadata')
    agent.restore('prev_agent')

# Start training
i = 0
max_episode_reward_mean = float('-inf')
while True:
    i += 1

    results = agent.train()

    episode_reward_mean = results['episode_reward_mean']

    # Print and log each iteration
    if not gcloud_run:
        print(pretty_print(results))
    with open('output_dir/train.log', 'a') as f:
        f.write('%d\t%f\n' % (i, episode_reward_mean))
    with open('output_dir/train_verbose.log', 'a') as f:
        f.write(pretty_print(results) + '\n')

    # Check for new best score
    if episode_reward_mean > max_episode_reward_mean:
        # First, save to local dir
        checkpoint = agent.save('output_dir')
        with open(os.path.join('output_dir', 'best_agents.log'), 'a') as f:
            f.write('%s -- %f\n' % (checkpoint, episode_reward_mean))

        if gcloud_run:
            # Now upload to cloud storage
            upload_blob(
                bucket_name=output_dir['bucket'],
                source_file_name=checkpoint + '.extra_data',
                destination_blob_name=os.path.join(output_dir['path'], os.path.basename(checkpoint) + '.extra_data'))
            upload_blob(
                bucket_name=output_dir['bucket'],
                source_file_name=checkpoint + '.tune_metadata',
                destination_blob_name=os.path.join(output_dir['path'], os.path.basename(checkpoint) + '.tune_metadata'))
            upload_blob(
                bucket_name=output_dir['bucket'],
                source_file_name=os.path.join('output_dir', 'best_agents.log'),
                destination_blob_name=os.path.join(output_dir['path'], 'best_agents.log'))

        # Update best score
        max_episode_reward_mean = episode_reward_mean

    # Periodic checkpointing to training logs
    # Also save agent checkpoint if you need it, but you'll have to manually download it
    if i % checkpoint_period == 0:
        #checkpoint = agent.save()
        #print('checkpoint saved at %s' % checkpoint)
        if gcloud_run:
            # Save training logs to cloud storage
            upload_blob(
                bucket_name=output_dir['bucket'],
                source_file_name='output_dir/train.log',
                destination_blob_name=os.path.join(output_dir['path'], 'train.log'))
            upload_blob(
                bucket_name=output_dir['bucket'],
                source_file_name='output_dir/train_verbose.log',
                destination_blob_name=os.path.join(output_dir['path'], 'train_verbose.log'))
        else:
            pass
            #checkpoint = agent.save()
            #print('checkpoint saved at %s' % checkpoint)
