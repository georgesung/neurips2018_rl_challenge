from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import pickle
from osim.env import ProstheticsEnv
from gym.spaces.box import Box

class ProstheticsEnvWrapper(object):
    def __init__(self,
                env,
                do_norm=True,
                state_norm_file=None,
                custom_reward=False,
                custom_reward_params=None,
                downsample_factor=None,
                reduce_action=False,
                test_mode=False,
                debug=False):
        # The basics
        self.env = env
        self.observation_space = Box(low=0, high=0, shape=(self.get_observation_space_size(),))  # custom
        self.reduce_action = reduce_action
        if self.reduce_action:
            self.reduced_action_dim = 7
            self.action_space = Box(low=-1., high=1., shape=(self.reduced_action_dim,), dtype=np.float32)
        else:
            self.action_space = self.env.action_space

        # Needed for RLLib to be used
        self.reward_range = (-np.inf, np.inf)
        self.metadata = {}

        # Other params
        self.do_norm = do_norm
        if do_norm:
            self.state_norm_const = pickle.load(open(state_norm_file, 'rb'))
        self.custom_reward = custom_reward
        if self.custom_reward:
            self.custom_reward_params = custom_reward_params
        self.test_mode = test_mode
        self.debug = debug

        self.downsample_factor = 1
        if downsample_factor is not None:
            self.downsample_factor = downsample_factor
        '''
        # Sim time limit and downsampling factor
        # downsample_factor = number of real time steps per visible time step
        self.time_limit = 300
        self.downsample_factor = 1
        if downsample_factor is not None:
            if self.time_limit % downsample_factor != 0:
                raise ValueError('time_limit mod downsample_factor != 0')
            self.downsample_factor = downsample_factor
        self.time_limit = self.time_limit / self.downsample_factor
        '''

    def dict_to_list(self, state_desc):
        """https://www.endtoend.ai/blog/ai-for-prosthetics-3/"""
        res = []

        # Body Observations
        for info_type in ['body_pos', 'body_pos_rot',
                          'body_vel', 'body_vel_rot',
                          'body_acc', 'body_acc_rot']:
            for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                              'femur_l', 'femur_r', 'head', 'pelvis',
                              'torso', 'pros_foot_r', 'pros_tibia_r']:
                res += state_desc[info_type][body_part]

        # Joint Observations
        # Neglecting `back_0`, `mtp_l`, `subtalar_l` since they do not move
        for info_type in ['joint_pos', 'joint_vel', 'joint_acc']:
            for joint in ['ankle_l', 'ankle_r', 'back', 'ground_pelvis',
                          'hip_l', 'hip_r', 'knee_l', 'knee_r']:
                res += state_desc[info_type][joint]

        '''
        # Don't care about force and muscles, that's directly correlated with actions
        # Muscle Observations
        for muscle in ['abd_l', 'abd_r', 'add_l', 'add_r', 
                       'bifemsh_l', 'bifemsh_r', 'gastroc_l',
                       'glut_max_l', 'glut_max_r', 
                       'hamstrings_l', 'hamstrings_r',
                       'iliopsoas_l', 'iliopsoas_r', 'rect_fem_l', 'rect_fem_r',
                       'soleus_l', 'tib_ant_l', 'vasti_l', 'vasti_r']:
            res.append(state_desc['muscles'][muscle]['activation'])
            res.append(state_desc['muscles'][muscle]['fiber_force'])
            res.append(state_desc['muscles'][muscle]['fiber_length'])
            res.append(state_desc['muscles'][muscle]['fiber_velocity'])

        # Force Observations
        # Neglecting forces corresponding to muscles as they are redundant with
        # `fiber_forces` in muscles dictionaries
        for force in ['AnkleLimit_l', 'AnkleLimit_r',
                      'HipAddLimit_l', 'HipAddLimit_r',
                      'HipLimit_l', 'HipLimit_r', 'KneeLimit_l', 'KneeLimit_r']:
            res += state_desc['forces'][force]

        # Center of mass position is only X and Y, no Z!
        # Center of Mass Observations
        res += state_desc['misc']['mass_center_pos']
        res += state_desc['misc']['mass_center_vel']
        res += state_desc['misc']['mass_center_acc']
        '''

        return res

    def norm_pos_to_pelvis(self, state_desc):
        """Normalize X, Z of other body parts to pelvis"""
        pelvis_x, _, pelvis_z = state_desc['body_pos']['pelvis']

        for body_part in ['calcn_l', 'talus_l', 'tibia_l', 'toes_l',
                          'femur_l', 'femur_r', 'head', 'pelvis',
                          'torso', 'pros_foot_r', 'pros_tibia_r']:
            state_desc['body_pos'][body_part][0] -= pelvis_x
            state_desc['body_pos'][body_part][2] -= pelvis_z

        return state_desc

    def process_state_desc(self, state_desc):
        state = np.array(self.dict_to_list(state_desc))

        # Add target_vel to state vector (round 2)
        #state = np.append(state, state_desc['target_vel'])

        if self.do_norm:
            # Also normalize target_vel (round 2)
            #state_norm_mu = np.append(self.state_norm_const['mu'], [0., 0., 0.])
            #state_norm_sigma = np.append(self.state_norm_const['sigma'], [3., 3., 3.])

            # For round 1
            state_norm_mu = self.state_norm_const['mu']
            state_norm_sigma = self.state_norm_const['sigma']

            state = (state - state_norm_mu) / state_norm_sigma

        return state

    def expand_action(self, action_reduced):
        '''
        Given reduced action space, expand action to original action space
        Reduced action space is organized as follows:
            [
                r_hip_add_abd, r_hip_flex_ext, r_knee_flex_ext,
                l_hip_add_abd, l_hip_flex_ext, l_knee_flex_ext,
                l_ankle_flex_ext
            ]
        Each value ranges from -1 to 1, where -1 is full effort adduction/flexion, +1 is full effort abduction/extension
        See https://goo.gl/images/gWEFWA for biomechanical explanation
        '''
        # Placeholder for target (original) action space
        action = [0. for _ in range(19)]

        # Right hip adduction/abduction
        reduced_val = action_reduced[0]
        action[1] = -min(reduced_val, 0.)  # right hip adductor
        action[0] =  max(reduced_val, 0.)  # right hip abductor

        # Right hip flexion/extension
        reduced_val = action_reduced[1]
        action[5] = -min(reduced_val, 0.)  # right hip flexor
        action[4] =  max(reduced_val, 0.)  # right glute

        # Right knee flexion/extension
        reduced_val = action_reduced[2]
        action[2] = -min(reduced_val, 0.)  # right hamstring
        action[3] = -min(reduced_val, 0.)  # right mini-hamstring
        action[6] =  max(reduced_val, 0.)  # right quad
        action[7] =  max(reduced_val, 0.)  # right mini-quad

        # Left hip adduction/abduction
        reduced_val = action_reduced[3]
        action[9] = -min(reduced_val, 0.)  # left hip adductor
        action[8] =  max(reduced_val, 0.)  # left hip abductor

        # Left hip flexion/extension
        reduced_val = action_reduced[4]
        action[13] = -min(reduced_val, 0.)  # left hip flexor
        action[12] =  max(reduced_val, 0.)  # left glute

        # Left knee flexion/extension
        reduced_val = action_reduced[5]
        action[10] = -min(reduced_val, 0.)  # left hamstring
        action[11] = -min(reduced_val, 0.)  # left mini-hamstring
        action[14] =  max(reduced_val, 0.)  # left quad
        action[15] =  max(reduced_val, 0.)  # left mini-quad

        # Left ankle flexion/extension
        reduced_val = action_reduced[6]
        action[18] = -min(reduced_val, 0.)  # left tibialis anterior (muscle on front of lower leg)
        action[16] =  max(reduced_val, 0.)  # left calf
        action[17] =  max(reduced_val, 0.)  # left mini-calf

        # Make action a list of floats (no numpy stuff)
        action = np.array(action).tolist()

        return action

    def customize_reward(self, next_state_desc, reward, terminal):
        if self.custom_reward_params['fall_penalty'] is not None:
            # Penalize "falling down", i.e. pelvis height < 0.6
            # But I'm not sure if it's <0.6 or <=0.6, just I'll use terminal signal too
            # Will be wrong if agent falls down at enforced sim time limit, but rare
            if next_state_desc['body_pos']['pelvis'][1] <= 0.6 and terminal:
                return self.custom_reward_params['fall_penalty']  # fall penalty, ignore all other rewards

        if self.custom_reward_params['orig_reward_floor'] is not None:
            # Floor/clip original reward
            reward = max(reward, self.custom_reward_params['orig_reward_floor'])

        if self.custom_reward_params['reward_offset'] is not None:
            # Add offset to reward, e.g. add positive number to encourage not falling
            reward += self.custom_reward_params['reward_offset']

        if self.custom_reward_params['hip_knee_flexion_reward'] is not None:
            # Encourage agent to bend knees and flex hip
            # Lots of fudge factors here!
            max_flexion_reward = 0.5
            hip_flexion_r  = min(next_state_desc['joint_pos']['hip_r'][0], max_flexion_reward)
            knee_flexion_r = min(-0.5 * next_state_desc['joint_pos']['knee_r'][0], max_flexion_reward)
            hip_flexion_l  = min(next_state_desc['joint_pos']['hip_l'][0], max_flexion_reward)
            knee_flexion_l = min(-0.5 * next_state_desc['joint_pos']['knee_l'][0], max_flexion_reward)
            hip_knee_flexion_reward = (hip_flexion_r + knee_flexion_r + hip_flexion_l + knee_flexion_l) / (4*max_flexion_reward)
            reward += hip_knee_flexion_reward

        if self.custom_reward_params['knee_flexion_reward'] is not None:
            # Encourage agent to bend knees
            # Note fudge factor here!
            max_flexion_reward = self.custom_reward_params['knee_flexion_reward']['max_flexion_reward']
            reward_scale = self.custom_reward_params['knee_flexion_reward']['reward_scale']
            knee_flexion_r = min(-reward_scale * next_state_desc['joint_pos']['knee_r'][0], max_flexion_reward)
            knee_flexion_l = min(-reward_scale * next_state_desc['joint_pos']['knee_l'][0], max_flexion_reward)
            knee_flexion_reward = knee_flexion_r + knee_flexion_l
            reward += knee_flexion_reward

        if self.custom_reward_params['head_pelvis_delta_x_reward'] is not None:
            # Encourage head x-position to be in front of pelvis
            head_pelvis_delta_x = next_state_desc['body_pos']['head'][0] - next_state_desc['body_pos']['pelvis'][0]
            head_pelvis_delta_x = min(head_pelvis_delta_x, 1.)  # don't reward too far lean forward
            reward += head_pelvis_delta_x

        if self.custom_reward_params['feet_center_pelvis_dist_penalty'] is not None:
            # Penalize distance b/w center of feet and pelvis, on XZ plane
            penalty_scale = self.custom_reward_params['feet_center_pelvis_dist_penalty']['penalty_scale']

            feet_center_x = (next_state_desc['body_pos']['talus_l'][0] + next_state_desc['body_pos']['pros_foot_r'][0]) / 2
            feet_center_z = (next_state_desc['body_pos']['talus_l'][2] + next_state_desc['body_pos']['pros_foot_r'][2]) / 2

            pelvis_x, _, pelvis_z = next_state_desc['body_pos']['pelvis']

            #feet_center_pelvis_dist = np.sqrt((feet_center_x - pelvis_x)**2 + (feet_center_z - pelvis_z)**2)
            feet_center_pelvis_dist_sq = (feet_center_x - pelvis_x)**2 + (feet_center_z - pelvis_z)**2
            feet_center_pelvis_dist_penalty = -penalty_scale * feet_center_pelvis_dist_sq

            reward += feet_center_pelvis_dist_penalty

        if self.custom_reward_params['exp_reward']:
            reward = math.exp(reward)  # WARNING: np.exp() is slower for only 1 value!

        return reward

    def reset(self):
        state_desc = self.env.reset(project=False)
        state = self.process_state_desc(state_desc)

        return state

    def step(self, action):
        if self.reduce_action:
            action = self.expand_action(action)

        total_reward = 0.  # only matters if self.downsample_factor > 1
        for _ in range(self.downsample_factor):
            next_state_desc, reward, terminal, info = self.env.step(action, project=False)
            next_state = self.process_state_desc(next_state_desc)
            if self.custom_reward:
                reward = self.customize_reward(next_state_desc, reward, terminal)

            total_reward += reward

            if self.debug:
                info['next_state_desc'] = next_state_desc

            if terminal:
                break

        # If we're in test mode we return total reward, otherwise return last reward
        # Only matters if downsample_factor > 1
        if self.test_mode:
            reward = total_reward

        return next_state, reward, terminal, info

    def get_action_space_size(self):
        if self.reduce_action:
            return self.reduced_action_dim
        else:
            return 19

    def get_observation_space_size(self):
        #return 389  # old, don't use
        return 249  # round 1
        #return 249+3  # +3 for round 2
