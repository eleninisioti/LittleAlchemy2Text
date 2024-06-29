

class Human():

    def __init__(self, idx, env, task_descript):
        self.idx = idx
        self.env = env
        self.done = False
        self.description= task_descript

        self.type = "human"

        self.env.reset()


    def move(self, state):
        repeat = True
        while repeat:
            item1 = input('Player ' + str(self.idx) + ', choose the first item: ')
            item2 = input('Player ' + str(self.idx) + ', choose the second item: ')

            if item1 in self.env.table and item2 in self.env.table:
                action = [int(self.env.table.index(item1)), int(self.env.table.index(item2))]
                repeat = False
            else:
                print("The items you chose are not in the inventory, choose again.")
        return action