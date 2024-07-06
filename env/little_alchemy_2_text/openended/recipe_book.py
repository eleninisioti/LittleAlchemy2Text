import os
import json
import pickle
import collections
import random
import timeit
import copy
import numpy as np
from gym.utils import seeding



class Recipe(collections.Counter):
    """A hashable recipe.
    Allows for indexing into dictionaries.
    """
    def __hash__(self):
        return tuple(
                sorted(
                    self.items(),
                    key=lambda x: x[0] if x[0] is not None else '')).__hash__()

    def __len__(self):
        return len(list(self.elements()))


class Task:
    """
    A hashable recipe task.
    """
    def __init__(self,  base_entities):
        self.base_entities = sorted(base_entities)

    def __hash__(self):
        return tuple((self.goal)).__hash__()


class RecipeBook:
    def __init__(self,
        data_path='datasets/alchemy2.json',split=None, train_ratio=1.0, seed=None):
        self.test_mode = False
        self.set_seed(seed)

        self._rawdata = self._load_data(data_path)

        self.entities = tuple(self._rawdata['entities'].keys())
        self.entity2index = {e:i for i,e in enumerate(self.entities)}
        self.entity2recipes = collections.defaultdict(list)

        for e in self.entities:
            for r in self._rawdata['entities'][e]['recipes']:
                if e not in r:
                    self.entity2recipes[e].append(Recipe(r))
        self.entity2recipes = dict(self.entity2recipes)

        self.max_recipe_size = 0
        self.recipe2entity = collections.defaultdict(str)
        for entity, recipes in self.entity2recipes.items():
            for r in recipes:
                self.recipe2entity[r] = entity
                self.max_recipe_size = max(len(r), self.max_recipe_size)



    def _random_choice(self, options):
        # Fast random choice
        i = self.np_random.integers(0, len(options))
        return options[i]

    def _load_data(self, path):
        f = open(path)
        jsondata = json.load(f)
        f.close()

        return jsondata

    def set_seed(self, seed):
        self.np_random, self.seed = seeding.np_random(seed)

    def save(self, path):
        """
        Serialize to bytes and save to file
        """
        path = os.path.expandvars(os.path.expanduser(path))
        f = open(path, 'wb+')
        pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
        Returns a new RecipeBook object loaded from a binary file that is the output of save.
        """
        path = os.path.expandvars(os.path.expanduser(path))
        f = open(path, 'rb')
        return pickle.load(f)

    def get_recipes(self, entity):
        return self.entity2recipes[entity] if entity in self.entity2recipes else None

    def evaluate_recipe(self, recipe):
        e = self.recipe2entity[recipe]
        return e if e != '' else None

    def init_neighbors_combineswith(self):
        self.neighbors_combineswith = collections.defaultdict(set)
        for recipe in self.recipe2entity:
            e1, e2 = recipe if len(recipe.keys()) == 2 else list(recipe.keys())*2
            self.neighbors_combineswith[e1].add(e2)
            self.neighbors_combineswith[e2].add(e1)

    def sample_task(self):
        base_root_entities = ['earth', 'air', 'fire', 'water']
        #random_root_entities = random.sample(self.entities, 8)
        root_entities = base_root_entities
        task = Task(base_entities=root_entities)
        return task
