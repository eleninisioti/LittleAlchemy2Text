import os
from enum import IntEnum

import numpy as np
import gym
from gym.utils import seeding
from gym.envs.registration import register

from env.wordcraft.wordcraft.env import WordCraftEnv


NO_RECIPE_PENALTY = 0
IRRELEVANT_RECIPE_PENALTY = -0.1
GOAL_REWARD = 1.0
SUBGOAL_REWARD = 1.0
from utils import seed as utils_seed
from utils.word2feature import FeatureMap
import random
import string
from env.little_alchemy_2_text.targeted.recipe_book import Recipe, RecipeBook


class LittleAlchemy2TextTargeted(WordCraftEnv):

    def __init__(self,
                 data_path='env/wordcraft/datasets/alchemy2.json',
                 recipe_book_path=None,
                 feature_type='glove',
                 shuffle_features=False,
                 random_feature_size=300,
                 max_depth=1,
                 split='by_recipe',
                 train_ratio=1.0,
                 num_distractors=0,
                 uniform_distractors=False,
                 max_mix_steps=1,
                 subgoal_rewards=True,
                 seed=None,
                 encoded=False
                 ):

        self.eval_mode = False

        if seed is None:
            seed = int.from_bytes(os.urandom(4), byteorder="little")
        self.set_seed(seed)
        utils_seed(seed)

        self.recipe_book = RecipeBook(
            data_path=data_path, max_depth=max_depth, split=split, train_ratio=train_ratio, seed=seed)

        self.feature_map = FeatureMap(
            words=self.recipe_book.entities,
            feature_type=feature_type, random_feature_size=random_feature_size,
            shuffle=shuffle_features,
            seed=seed)

        self.max_selection_size = self.recipe_book.max_recipe_size
        self.max_mix_steps = max(max_mix_steps or max_depth, max_depth)
        self.max_steps = self.max_selection_size * self.max_mix_steps

        self.sample_depth = max_depth

        self.subgoal_rewards = subgoal_rewards
        self.max_depth = max_depth
        self.num_distractors = num_distractors
        self.uniform_distractors = uniform_distractors

        self.max_table_size = 2 ** max_depth + num_distractors + self.max_mix_steps

        self.task = None
        self.distractors = []
        self.goal_features = np.zeros(self.feature_map.feature_dim)

        self._reset_table()
        self._reset_selection()
        self._reset_history()

        self.episode_step = 0
        self.episode_mix_steps = 0
        self.episode_reward = 0
        self.done = False

        obs = self.reset()
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
        self.action_space = gym.spaces.Discrete(self.max_table_size)


        self.success = False
        self.encoded = encoded

    def reset(self):
        self.past_invalid_combs = []
        self.past_valid_combs = {}

        self.episode_step = 0
        self.episode_mix_steps = 0
        self.episode_reward = 0
        self.done = False


        self.task = self.recipe_book.sample_task()
        self.distractors = self.recipe_book.sample_distractors(self.task, self.num_distractors,
                                                               uniform=self.uniform_distractors)

        self._reset_selection()
        self._reset_table()
        self._reset_history()

        return self._get_observation()


    def _reset_history(self):
        self.subgoal_history = {}



    def step(self, actions):
            # ---- first mixing step ----
            reward = 0
            if self.done:  # no-op if env is done

                return self._get_observation(), reward, self.done, {}

            # Handle invalid actions
            for action in actions:
                invalid_action = not (0 <= action < self.max_table_size)

            if invalid_action:
                print("invalid action inside env")
                self.episode_step += 1
                if self.episode_step >= self.max_steps:
                    self.done = True

            # first word
            action = actions[0]
            i = self.table_index[action]
            e = self.recipe_book.entities[i]
            if self.encoded:
                new_comb = "'" + self.encode(e) + "'"

            else:
                new_comb = "'" + e + "'"

            selection_size = len(self.selection)
            self.selection.append(e)
            self.selection_index[selection_size] = i
            self.selection_features[selection_size, :] = self.feature_map.feature(e)
            selection_size = len(self.selection)

            # second word
            action = actions[1]
            self.episode_mix_steps += 1
            i = self.table_index[action]
            e = self.recipe_book.entities[i]
            if self.encoded:
                new_comb += " and '" + self.encode(e) + "'"

            else:
                new_comb += " and '" + e + "'"

            selection_size = len(self.selection)
            self.selection.append(e)
            self.selection_index[selection_size] = i
            self.selection_features[selection_size, :] = self.feature_map.feature(e)

            # Evaluate selection
            selection = self.selection
            recipe = Recipe(self.selection)
            result = self.recipe_book.evaluate_recipe(recipe)

            if result is None:
                reward = 0
            else:

                if result not in self.table:
                    self.subgoal_history[new_comb] = result
                    reward = 1

                    result_i = self.recipe_book.entity2index[result]
                    table_size = len(self.table)
                    self.table.append(result)
                    self.table_index[table_size] = result_i
                    self.table_features[table_size, :] = self.feature_map.feature(result)

            self.episode_reward += reward

            # Clear selection
            self._reset_selection()

            self.episode_step += 1
            if (self.episode_mix_steps >= self.max_mix_steps or self.episode_step >= self.max_steps):
                self.done = True

            if result == self.task.goal:
                self.done = True

            obs = self._get_observation()
            self.remaining_rounds = self.max_mix_steps - self.episode_step

            info = {"remaining_rounds": self.remaining_rounds}

            if not reward:
                if actions not in self.past_invalid_combs:
                    self.past_invalid_combs.append(tuple(actions))
            if result:
                if tuple(actions) not in self.past_valid_combs.keys():
                    self.past_valid_combs[tuple(actions)] = self.table.index(result)
            return obs, reward, self.done, info

    def encode(self, word):
        length = 5
        random.seed(word)
        letters = string.ascii_lowercase
        return ''.join(random.choice(letters) for i in range(length))

    def index_to_word(self, index):
        return self.table[index]

    def word_to_index(self, word):
        items = list(self.recipe_book.entities)
        if self.encoded:
            items = [self.encode[el] for el in items]
        return items[word]


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

    def render(self, mode='human'):
        return self._display_llm()

    def invalid_combs_to_string(self, past_invalid_combs):
        past_invalid_combs_str = ""
        for element in past_invalid_combs:
            past_invalid_combs_str += '"' + str(self.index_to_word(element[0])) + '" and "' + str(
                self.index_to_word(element[1])) + '", '
        return past_invalid_combs_str, len(past_invalid_combs)

    def get_invalid_combs(self):
        "Returns invalid combinations as a string"
        if self.encoded:
            past_invalid_combs = [self.encode(el) for el in self.past_invalid_combs]

        else:
            past_invalid_combs = self.past_invalid_combs

        return past_invalid_combs[-15:], len(past_invalid_combs[-15:])

    def valid_combs_to_string(self, past_valid_combs):
        valid_combs = ""

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
        return valid_combs, len(past_valid_combs)

    def get_valid_combs(self):
        "Returns invalid combinations as a string"
        past_valid_combs = list(self.past_valid_combs.keys())

        return past_valid_combs, len(past_valid_combs)


register(
    id='LittleAlchemy2TextTargeted-v0',
    entry_point='env.little_alchemy_2_text.targeted.env:LittleAlchemy2TextTargeted',
)