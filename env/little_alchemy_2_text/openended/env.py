""" Contains the implementation of an open-ended task.
"""

import gym
from env.little_alchemy_2_text.openended.recipe_book import Recipe, RecipeBook
from gym.envs.registration import register
from env.little_alchemy_2_text.base import LittleAlchemy2Text

NO_RECIPE_PENALTY = -0.1
IRRELEVANT_RECIPE_PENALTY = -0.1
GOAL_REWARD = 1.0
SUBGOAL_REWARD = 1.0


class LittleAlchemy2TextOpen(LittleAlchemy2Text):

    def __init__(self,
                 seed,
                 encoded=False,
                 max_mix_steps=1):

        super().__init__(seed=seed, max_mix_steps=max_mix_steps, encoded=encoded)

        self.num_distractors = 0
        self.max_depth = 1

        self.recipe_book = RecipeBook(
            data_path=self.data_path,
            seed=self.seed)

        self._setup(self.recipe_book)

        num_entities = len(self.recipe_book.entities)

        dspaces = {
            'table_index': gym.spaces.MultiDiscrete(self.max_table_size * [num_entities]),
            'table_features': gym.spaces.Box(shape=self.table_features.shape, low=-1., high=1.),
            'selection_index': gym.spaces.MultiDiscrete(self.max_selection_size * [num_entities]),
            'selection_features': gym.spaces.Box(shape=self.selection_features.shape, low=-1., high=1.),
        }
        self.observation_space = gym.spaces.Dict(dspaces)

    def reset(self):
        self.distractors = []
        return super().reset()

    def _get_observation(self):
        return super()._get_observation()

    def step(self, actions):



        selection, new_comb, actions = self._parse_actions(actions)

        if selection is None:
            return self._get_observation(), None, None, {"repeat": True}
        else:
            recipe = Recipe(selection)

            _, obs, reward, self.done, info = super()._step(recipe, new_comb, actions)
            info["repeat"] = False

            return obs, reward, self.done, info

    def summarise(self):
        return " having " + str(len(self.table)) + " items in their inventory"

    def _display_llm(self):
        inventory = self.table

        if self.encoded:
            inventory = [self.encode(el) for el in inventory]

        valid_combs, past_invalid_combs = self._print_valid_and_invalid_combs()

        output = "\n<human> INPUT \n Inventory: '" + "', '".join(inventory) + "'"
        output += "\nTask valid combinations (do not repeat combinations here): " + valid_combs
        output += "\nTask invalid combinations (do not repeat combinations here): " + ", ".join(past_invalid_combs)
        return output


register(
    id='LittleAlchemy2TextOpen-v0',
    entry_point='env.little_alchemy_2_text.openended.env:LittleAlchemy2TextOpen',
)
