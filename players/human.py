
from flask import Flask, request, jsonify, render_template

class Human():

    def __init__(self, idx, env, task_descript):
        self.idx = idx
        self.env = env
        self.done = False
        self.description= task_descript

        self.type = "human"

        self.env.reset()

    def move(self, state):
        item1 = input('Player ' + str(self.idx) + ', choose the first item: ')
        item2 = input('Player ' + str(self.idx) + ', choose the second item: ')

        action = "Combination: '" + item1 + "' and '" + item2 + "'"
        return action