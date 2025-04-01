""" A human player is asked to combine two items in each round through the command line.
"""


class Human():

    def __init__(self, idx, env, task_descript, seed):
        self.idx = idx # id in the group
        self.type = "human"

        self.env = env
        self.done = False
        self.description = task_descript

        self.env.reset(seed=seed)

    def move(self, state):
        item1 = input('Player ' + str(self.idx) + ', you may choose the first item: ')
        item2 = input('Player ' + str(self.idx) + ', you may choose the second item: ')
        action = "Combination: '" + item1 + "' and '" + item2 + "'"
        return action
