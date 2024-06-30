import os
from enum import IntEnum

import numpy as np
import gym
from gym.utils import seeding
from gym.envs.registration import register


NO_RECIPE_PENALTY = 0
IRRELEVANT_RECIPE_PENALTY = -0.1
GOAL_REWARD = 1.0
SUBGOAL_REWARD = 1.0
from utils import seed as utils_seed
from utils.word2feature import FeatureMap
import random
import string
from env.little_alchemy_2_text.targeted.recipe_book import Recipe, RecipeBook
from env.little_alchemy_2_text.base import LittleAlchemy2Text

class LittleAlchemy2TextTargeted(LittleAlchemy2Text):

    def __init__(self,
                 seed,
                 max_mix_steps=1,
                 encoded=False,
                 max_depth=1,
                 split='by_recipe',
                 train_ratio=1.0,
                 num_distractors=0,
                 ):

        super().__init__(seed, encoded, max_mix_steps)

        self.num_distractors = num_distractors
        self.max_depth = max_depth

        self.recipe_book = RecipeBook(
            data_path=self.data_path, max_depth=max_depth, split=split, train_ratio=train_ratio, seed=seed)

        self._setup(self.recipe_book)

        num_entities = len(self.recipe_book.entities)

        dspaces = {
            'goal_index': gym.spaces.MultiDiscrete([num_entities]),
            'goal_features': gym.spaces.Box(shape=self.goal_features.shape, low=-1., high=1.),
            'table_index': gym.spaces.MultiDiscrete(self.max_table_size * [num_entities]),
            'table_features': gym.spaces.Box(shape=self.table_features.shape, low=-1., high=1.),
            'selection_index': gym.spaces.MultiDiscrete(self.max_selection_size * [num_entities]),
            'selection_features': gym.spaces.Box(shape=self.selection_features.shape, low=-1., high=1.),
        }
        self.observation_space = gym.spaces.Dict(dspaces)



    def reset(self):
        super().reset()
        self.distractors = self.recipe_book.sample_distractors(self.task, self.num_distractors,
                                                               uniform=self.uniform_distractors)

        self._reset_table()
        return self._get_observation()

    def _get_observation(self):
        """
        Note, includes indices for each inventory and selection item,
        since torchbeast stores actions in a shared_memory tensor shared among actor processes
        """
        return {
            'goal_index': [self.recipe_book.entity2index[self.task.goal]],
            'goal_features': self.goal_features,
            'table_index': self.table_index,
            'table_features': self.table_features,
            'selection_index': self.selection_index,
            'selection_features': self.selection_features,
        }

    def step(self, actions):

        selection, new_comb, actions = self._parse_actions(actions)

        if selection is None:
            return self._get_observation(), None, None, {"repeat": True}
        else:
            recipe = Recipe(selection)

            result, obs, reward, self.done, info = super()._step(recipe, new_comb, actions)
            info["repeat"] = False

            if result == self.task.goal:
                self.done = True

            return obs, reward, self.done, info

    def summarise(self):
        if self.done:
            return " having discovered the target in " + str(self.episode_step) + " rounds"
        else:
            return " not having discover the target."

    def _display_llm(self):
        inventory = self.table
        target = self.task.goal

        if self.encoded:
            inventory = [self.encode(el) for el in inventory]
            target = self.encode(target)
        remaining_rounds = self.max_mix_steps - self.episode_step
        valid_combs = ""
        counter = 0
        for key, val in self.past_valid_combs.items():
            subkeys = []
            for subkey in key:

                subkeys.append(str(self.index_to_word(subkey)))
            new_key = '"' + subkeys[0] + '" and "' + subkeys[1]
            val = str(self.index_to_word(val))
            if self.encoded:
                valid_combs += new_key + " -> " + self.encode(val) + " , "
            else:
                valid_combs += new_key + " -> " + val + " , "

            counter = counter + 1
            if counter > 15:
                break

        #if self.encoded:
        #    past_invalid_combs = [self.encode(el) for el in self.past_invalid_combs]

        #else:
        #    past_invalid_combs = self.past_invalid_combs
        past_invalid_combs = self.past_invalid_combs
        past_invalid_combs = past_invalid_combs[-15:]
        past_invalid_combs_str = []
        for el in past_invalid_combs:
            past_invalid_combs_str.append('"' + str(self.index_to_word(el[0])) + '" and "' + str(self.index_to_word(el[1])) + '"')

        if self.encoded:
            self.env.env.table = [self.encode(el) for el in self.env.env.table]


        output = "\n<human> INPUT \n Inventory: '" + "', '".join(inventory) + "'"
        output += "\nTarget: '" + str(target) + "'"
        output += "\nRemaining rounds: " + str(remaining_rounds)
        output += "\nNumber of intermediate items: " + str(len(self.task.intermediate_entities))
        output += "\nTask valid combinations (do not repeat combinations here): " + valid_combs
        output += "\nTask invalid combinations (do not repeat combinations here): " + ", ".join(past_invalid_combs_str)
        return output




register(
    id='LittleAlchemy2TextTargeted-v0',
    entry_point='env.little_alchemy_2_text.targeted.env:LittleAlchemy2TextTargeted',
)