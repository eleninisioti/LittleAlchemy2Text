import sys
import os
sys.path.append(".")
sys.path.append("env/wordcraft")

import argparse
import env.little_alchemy_2_text.openended.env
import env.little_alchemy_2_text.targeted.env

import gym
def game(args):

    print("New game starts. \n")

    if args.targeted:
        description = "Combine the available items to make the target item"
        env = gym.make("LittleAlchemy2TextTargeted-v0",
                       seed=args.seed,
                       max_mix_steps= args.rounds,
                       num_distractors=args.distractors,
                       max_depth=args.depth,
                       encoded=args.encoded)
    else:
        description = "Combine the available items to make as many items as possible."
        env = gym.make("LittleAlchemy2TextOpen-v0",
                       seed=args.seed,
                       max_mix_steps= args.rounds,
                       encoded=args.encoded)
    print(description)

    env.reset()

    state = env.render()
    print(state)

    for i in range(args.rounds):

        repeat = True
        while repeat:
            item1 = input('Choose the first item: ')
            item2 = input('Choose the second item: ')

            if item1 in env.table and item2 in env.table:
                action = [int(env.table.index(item1)), int(env.table.index(item2))]
                repeat = False
            else:
                print("The items you chose are not in the inventory, choose again.")

        obs, reward, done, info = env.step(action)

        if done:
            print("Nice! You found the target item in " + str(i+1) + " rounds")
        state = env.render()
        print(state)

    print("Nice! Your inventory has " + str(len(env.table)) + " items.")
    print("Game ended")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LittleAlchemy2Text is a text-based version of the game Little Alchemy 2')

    parser.add_argument('-N', '--nplayers', type=int, default=1, help='Number of players')
    parser.add_argument('-t', '--targeted', action='store_true', help='If true, task is targeted. Otherwise, the task is openended')
    parser.add_argument('-d', '--distractors', type=int,default=3, help='Number of distractors for targetd tasks')
    parser.add_argument('-de', '--depth', type=int,default=1, help='Depth for targeted tasks')

    parser.add_argument('-r', '--rounds', type=int, default=10, help='Number of crafting rounds in a single game')
    parser.add_argument('-s', '--seed', type=int, default=0, help='Seed for the task')
    parser.add_argument('-e', '--encoded', action='store_true', help="Encode the words into random strings.")

    args = parser.parse_args()

    game(args)



