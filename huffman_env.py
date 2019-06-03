import numpy as np
from garage.envs.env_spec import EnvSpec
import pdb
import gym
import pickle
from garage.core import Serializable
from garage.tf.spaces.box import Box
from garage.tf.spaces.discrete import Discrete
from garage.envs.base import Step
from huffman import HuffmanCoding

from gym.spaces import Box as GymBox
from gym.spaces import Dict as GymDict
from gym.spaces import Discrete as GymDiscrete
from gym.spaces import Tuple as GymTuple

import pdb

class HuffmanEnv(gym.Env, Serializable):
    def __init__(self, data_file, parsed_file, num_classes):
        self.dataset = None
        self.parsed = None
        self.num_classes = num_classes
        with open(data_file, 'rb') as f:
            self.dataset = pickle.load(f)

        with open(parsed_file, 'rb') as f:
            self.parsed = pickle.load(f)
        if self.dataset is None or self.parsed is None:
            print("ERROR: COULD NOT LOAD DATASET")
            exit()

        # low = np.zeros((self.dataset.shape[1], self.dataset.shape[2]))
        # high = np.ones((self.dataset.shape[1], self.dataset.shape[2]))
        low = np.zeros((self.dataset.shape[1]))
        high = np.ones((self.dataset.shape[1]))
        self.access_array = np.arange(self.dataset.shape[0])
        self.observation_space_obj = GymBox(low=low, high=high)

        self.action_space_obj = GymDiscrete(num_classes)

        self.classes = np.zeros((self.dataset.shape[0]))
        # pdb.set_trace()
        Serializable.quick_init(self, locals())

    def step(self, action):
        self.classes[self.access_array[self.index]] = action
        self.index += 1

        reward = 0
        done = False
        obs = None

        if self.index >= self.dataset.shape[0]:
            #All examples classified, find reward
            print(self.classes)
            for j in range(self.num_classes):
                self.class_data = []
                for i in np.argwhere(self.classes == j).reshape(-1).tolist():
                    self.class_data += self.parsed[i*(self.dataset.shape[1]//16):(i+1)*(self.dataset.shape[1]//16)]
                if len(self.class_data) > 0:
                    h = HuffmanCoding('DJIEncoded.txt')
                    h.create_coding_from_binary(self.class_data)
                    encoded_array, size = h.get_encoded_array(self.class_data)
                    reward += size
            done = True
        else:
            obs = self.dataset[self.access_array[self.index],...]

        return Step(observation=obs,
                    reward=reward,
                    done=done,
                    info=None)

    def reset(self):
        self.index = 0
        np.random.shuffle(self.access_array)

        return self.dataset[self.access_array[self.index],...]

    @property
    def action_space(self):
        return self.action_space_obj

    @property
    def observation_space(self):
        return self.observation_space_obj



