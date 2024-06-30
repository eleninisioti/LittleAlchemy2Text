import sys
import os
import gym
from flask import Flask, request, jsonify, render_template

sys.path.append(os.getcwd())
sys.path.append("env/wordcraft")

from players.LLM import LLM

import env.little_alchemy_2_text.openended.env
import env.little_alchemy_2_text.targeted.env
from box import Box

app = Flask(__name__)

# Global variable to hold the environment
game_state = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start():
    global game_state

    args = {
        'targeted': False,
        'distractors': 3,
        'depth': 1,
        'rounds': 5,
        'seed': 5,
        'encoded': False
    }

    args = Box(args)

    env = gym.make("LittleAlchemy2TextOpen",
                   seed=args.seed,
                   max_mix_steps=args.rounds,
                   encoded=args.encoded)

    env_LLM = gym.make("LittleAlchemy2TextOpen",
                       seed=args.seed,
                       max_mix_steps=args.rounds,
                       encoded=args.encoded)

    env.reset()
    env_LLM.reset()
    state = env.render([env_LLM])

    # Save the environment for future actions
    game_state['env'] = env
    game_state['state'] = state
    state_LLM = env_LLM.render([env])
    llm_player = LLM(idx=0, env=env_LLM, targeted=args.targeted, multiagent=0)

    game_state['env_LLM'] = env_LLM
    game_state['state_LLM'] = state_LLM
    game_state['player_LLM'] = llm_player

    additional_info = state  # This is the initial state for the LLM player
    start_game_message = "LLM ready to play"

    return jsonify(state=state, additionalInfo=additional_info, startGameField=start_game_message)

@app.route('/update', methods=['POST'])
def update():
    global game_state

    if 'env' not in game_state:
        return jsonify(error="Game not started"), 400

    # Get the user input
    first_item = request.json.get('firstItem')
    second_item = request.json.get('secondItem')

    # Retrieve the environment from the global state
    env = game_state['env']

    # Example action based on user input
    action = f"Combination: '{first_item}' and '{second_item}'"

    obs, reward, done, info = env.step(action)
    env_LLM = game_state['env_LLM']


    if done:
        updated_state = env.summarise()
    else:
        updated_state = env.render([env_LLM])  # Update this with the actual game state change

    # Save the updated state
    game_state['state'] = updated_state

    # LLM player interaction
    state_LLM = game_state['state_LLM']
    player_LLM = game_state["player_LLM"]

    action = player_LLM.move(state_LLM)
    obs, reward, done, info = env_LLM.step(action)

    if done:
        additional_message = env_LLM.summarise()
    else:
        additional_message = env_LLM.render([env])  # Update this with the actual game state change

    # Additional messages
    start_game_message = additional_message

    return jsonify(state=updated_state, additionalMessage=start_game_message, startGameField=action)

if __name__ == '__main__':
    app.run(debug=True)
