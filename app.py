import sys
import os
import gym
from flask import Flask, request, jsonify, render_template, make_response
from flask_cors import CORS
sys.path.append(os.getcwd())
sys.path.append("env/wordcraft")
import time
from players.LLM import LLM

import env.little_alchemy_2_text.openended.env
import env.little_alchemy_2_text.targeted.env
from box import Box
import numpy
app = Flask(__name__)
CORS(app)  # Allow CORS
import pickle
# Global variable to hold the environment
game_states = {}

@app.route('/', methods=['GET'])
def index():
    response = make_response(render_template('index.html'))
    response.headers['Cache-Control'] = 'no-store, no-cache, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    return response

@app.route('/start', methods=['POST'])
def start():
    global game_states

    current_time_seconds = time.time()
    random_number = numpy.random.randint(1000, 10000)
    ID = str(random_number) + str(current_time_seconds)
    game_state = {}


    seed = 0
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
                   max_mix_steps=args.rounds,
                   encoded=args.encoded)

    env_LLM = gym.make("LittleAlchemy2TextOpen",
                       max_mix_steps=args.rounds,
                       encoded=args.encoded)

    env.reset(seed=seed)
    env_LLM.reset(seed=seed)
    state = env.render([env_LLM], "human")

    # Save the environment for future actions
    game_state['env'] = env
    game_state['state'] = state
    state_LLM = env_LLM.render([env])
    llm_player = LLM(idx=0, env=env_LLM, targeted=args.targeted, multiagent=0, seed=seed)

    game_state['env_LLM'] = env_LLM
    game_state['state_LLM'] = state_LLM
    game_state['player_LLM'] = llm_player

    additional_info = state  # This is the initial state for the LLM player
    start_game_message = ""

    game_states[ID] = game_state

    #with open("temp.pkl", "wb") as f:
    #pickle.dump()
    print(game_states)


    return jsonify(state=state, ID=ID, startGameField=start_game_message, additionalInfo=additional_info)

@app.route('/update', methods=['POST'])
def update():
    global game_states
    print(game_states)
    game_id = request.args.get('gameID')
    game_state = game_states[game_id]

    if 'env' not in game_state:
        return jsonify(error="Game not started"), 400

    first_word = request.json.get('userInput1')
    second_word = request.json.get('userInput2')

    action = "Combination: '" + first_word + "' and '" + second_word + "'"

    env = game_state['env']

    obs, reward, done, info = env.step(action)
    env_LLM = game_state['env_LLM']

    if done:
        human = env.summarise()
    else:
        human = env.render([env_LLM], "human")

    game_state['state'] = human

    state_LLM = game_state['state_LLM']
    player_LLM = game_state["player_LLM"]

    action = player_LLM.move(state_LLM)
    obs, reward, done, info = env_LLM.step(action)

    if done:
        llm = env_LLM.summarise()
    else:
        llm = env_LLM.render([env])

    llm += "\n" + action

    return jsonify(humanInfo=human, llmInfo=llm)

if __name__ == '__main__':
    app.run(debug=True)