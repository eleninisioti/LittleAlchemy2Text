""" An example script that you can use to play LittleAlchemy2Text with humans and LLMs.
 You can run it directly or call method 'play' from another script.

 A game consists of a group of (human and/or LLM) players attempting combinations in the same task one after each other.
 Players in a group do not share inventories but can observe the valid and invalid combinations of others.
 """

import sys
import os
sys.path.append(os.getcwd())
sys.path.append("env/wordcraft")
import env.little_alchemy_2_text.openended.env
import env.little_alchemy_2_text.targeted.env
import argparse
import gym
from players.human import Human
from players.LLM import LLM


def setup(args):
    nhuman = int(input('How many human players are there?'))
    nLLM = int(input('How many LLM players are there?'))

    group = []
    for i in range(nhuman):
        if args.targeted:
            task_descript = "Combine the available items to make the target item"
            env = gym.make("LittleAlchemy2TextTargeted-v0",
                           seed=args.seed,
                           max_mix_steps=args.rounds,
                           num_distractors=args.distractors,
                           max_depth=args.depth,
                           encoded=args.encoded)
        else:
            task_descript = "Combine the available items to make as many items as possible."
            env = gym.make("LittleAlchemy2TextOpen-v0",
                           seed=args.seed,
                           max_mix_steps=args.rounds,
                           encoded=args.encoded)
        group.append(Human(i, env, task_descript))

    for i in range(nhuman, nhuman + nLLM):
        if args.targeted:
            env = gym.make("LittleAlchemy2TextTargeted-v0",
                           seed=args.seed,
                           max_mix_steps=args.rounds,
                           num_distractors=args.distractors,
                           max_depth=args.depth,
                           encoded=args.encoded)
        else:
            env = gym.make("LittleAlchemy2TextOpen-v0",
                           seed=args.seed,
                           max_mix_steps=args.rounds,
                           encoded=args.encoded)

        group.append(LLM(i, env, targeted=args.targeted, multiagent=(nLLM - 1)))

    return group


def game(args):
    group = setup(args)

    print("New game starts. \n")

    for i in range(args.rounds):

        game_done = all([player.done for player in group])
        if not game_done:

            print("New round. Players: " + ' '.join([str(player.idx) for player in group if not player.done]))

            for player in group:

                if not player.done:

                    if player.type == "human":
                        print(player.description)

                    other_envs = [other_player.env for other_player in group if other_player.idx != player.idx]

                    state = player.env.render(other_envs)
                    print(state)

                    repeat = True

                    while repeat:
                        action = player.move(state)

                        obs, reward, done, info = player.env.step(action)
                        repeat = info["repeat"]

                    if done:
                        print("Nice! Player " + str(player.idx) + ", you found the target item in " + str(
                            i + 1) + " rounds")
                        player.done = True

    print("Nice! The game ended with: \n")
    for player in group:
        print("Player " + str(player.idx) + player.env.summarise())


def play(targeted=True, distractors=3, depth=1, rounds=5, seed=5):


    args = {}
    args['targeted'] = targeted
    args['distractors'] = distractors
    args['depth'] = depth
    args['rounds'] = rounds
    args['seed'] = seed

    game(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='LittleAlchemy2Text is a text-based version of the game Little Alchemy 2.')

    parser.add_argument('-t', '--targeted', action='store_true',
                        help='If true, task is targeted. Otherwise, the task is openended')
    parser.add_argument('-d', '--distractors', type=int, default=3, help='Number of distractors for targetd tasks')
    parser.add_argument('-de', '--depth', type=int, default=1, help='Depth for targeted tasks')
    parser.add_argument('-r', '--rounds', type=int, default=10, help='Number of crafting rounds in a single game')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Seed for the task')

    args = parser.parse_args()

    game(args)
