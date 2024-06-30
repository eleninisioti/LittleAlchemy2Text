
import os
from utils import seed as utils_seed
import gym
from utils.word2feature import FeatureMap
import numpy as np
from gym.utils import seeding
from env.wordcraft.wordcraft.env import WordCraftEnv

def find_nth(haystack, needle, n):
    """ Find the nth occurrence of sub-string in string.
     """
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start + len(needle))
        n -= 1
    return start

class LittleAlchemy2Text(WordCraftEnv):

    def __init__(self,
                 seed=0,
                 encoded=False,
                 max_mix_steps=1):

        self.feature_type = 'glove'
        self.shuffle_features = False
        self.random_feature_size=300
        self.uniform_distractors=False
        self.eval_mode = False
        self.data_path = "env/wordcraft/datasets/alchemy2.json"

        self.max_mix_steps = max_mix_steps
        self.seed = seed


        if seed is None:
            seed = int.from_bytes(os.urandom(4), byteorder="little")
        self.set_seed(seed)
        utils_seed(seed)

        self.success = False
        self.encoded = encoded

    def _setup(self, recipe_book):

        self.feature_map = FeatureMap(
            words=self.recipe_book.entities,
            feature_type=self.feature_type,
            random_feature_size=self.random_feature_size,
            shuffle=self.shuffle_features,
            seed=self.seed)

        self.max_selection_size = self.recipe_book.max_recipe_size
        self.max_steps = self.max_selection_size * self.max_mix_steps

        self.max_table_size = 2 ** self.max_depth + self.num_distractors + self.max_mix_steps

        self.task = None
        self.distractors = tuple([])
        self.goal_features = np.zeros(self.feature_map.feature_dim)

        self._reset_table()
        self._reset_selection()
        self._reset_history()

        self.episode_step = 0
        self.episode_mix_steps = 0
        self.episode_reward = 0
        self.done = False

        self.action_space = gym.spaces.Discrete(
            self.max_table_size)  # Actions correspond to choosing an entity in a table position

    def reset(self):
        self.past_invalid_combs = []
        self.past_valid_combs = {}

        self.episode_step = 0
        self.episode_mix_steps = 0
        self.episode_reward = 0
        self.done = False

        self.task = self.recipe_book.sample_task()

        self._reset_selection()
        self._reset_table()
        self._reset_history()

        return self._get_observation()


    def _reset_history(self):
        self.subgoal_history = {}

    def _get_observation(self):
        """
        Note, includes indices for each inventory and selection item,
        since torchbeast stores actions in a shared_memory tensor shared among actor processes
        """
        return {
            'table_index': self.table_index,
            'table_features': self.table_features,
            'selection_index': self.selection_index,
            'selection_features': self.selection_features,
        }

    def _string_to_actions(self, actions):

        start_first = find_nth(actions, "Combination: '", 1)
        actions = actions[start_first:]
        start_first = find_nth(actions, "Combination: '", 1)
        end_first = find_nth(actions, "'", 2)
        first_word = actions[start_first+len("Combination: '"):end_first]

        end_second = find_nth(actions, "'", 4)
        second_word = actions[(end_first+7): end_second]

        if first_word in self.table and second_word in self.table:
            action = [int(self.table.index(first_word)), int(self.table.index(second_word))]
            repeat = False
        else:
            print("The items you chose are not in the inventory, choose again.")
            action = None
            repeat = True
        return action, repeat

    def _parse_actions(self, actions):

        actions, repeat = self._string_to_actions(actions)

        if repeat:
            return None, None, None

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
        return selection, new_comb, actions

    def _step(self, recipe, new_comb,  actions):
        result = self.recipe_book.evaluate_recipe(recipe)

        if result is None:
            reward = 0
        else:
            reward = 0

            if result not in self.table:
                self.subgoal_history[new_comb] = result
                reward = 1

                result_i = self.recipe_book.entity2index[result]
                table_size = len(self.table)
                self.table.append(result)
                self.table_index[table_size] = result_i
                self.table_features[table_size, :] = self.feature_map.feature(result)


        # Clear selection
        self._reset_selection()

        self.episode_step += 1
        if (self.episode_mix_steps >= self.max_mix_steps or self.episode_step >= self.max_steps):
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
        return result, obs, reward, self.done, info

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
        return items.index(word)

    def render(self, envs, mode='human'):

        info = self._display_llm()

        if len(envs):
            social_info = self._display_social(envs)
            info = info + "\n" + social_info
        return info


    def _display_social(self, envs):
        social_info = "Other players valid combinations: "
        total_valid_combs = ""
        total_past_invalid_combs = ""

        for env in envs:
            valid_combs = ""

            counter = 0
            for key, val in env.past_valid_combs.items():
                subkeys = []
                for subkey in key:
                    subkeys.append(str(env.index_to_word(subkey)))
                new_key = "'" + subkeys[0] + "' and '" + subkeys[1] + "'"
                val = str(env.index_to_word(val))
                if env.encoded:
                    valid_combs += new_key + " -> '" + self.encode(val) + "' , "
                else:
                    valid_combs += new_key + " -> '" + val + "' , "

                counter = counter + 1
                if counter > 15:
                    break
            total_valid_combs += valid_combs

            past_invalid_combs = env.past_invalid_combs
            past_invalid_combs = past_invalid_combs[-15:]
            past_invalid_combs_str = []
            for el in past_invalid_combs:
                past_invalid_combs_str.append(
                    '"' + str(env.index_to_word(el[0])) + '" and "' + str(env.index_to_word(el[1])) + '"')
            total_past_invalid_combs += ", ".join(past_invalid_combs_str)

        social_info += total_valid_combs + "\nOther players invalid combinations (do not repeat combinations here): " + total_past_invalid_combs

        return social_info


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



    def summarise(self):
        pass
    def _display_llm(self):
        pass





