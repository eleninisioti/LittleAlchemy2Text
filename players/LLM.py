""" An LLM player is provided with a prompt describing the task and the current state of the environment.
"""

import ollama


class LLM:

    def __init__(self, idx, env, targeted, multiagent):
        self.idx = idx  # id the group
        self.env = env
        self.done = False

        self.setup(targeted, multiagent)

        self.type = "LLM"
        self.env.reset()

    def setup(self, targeted, multiagent):
        if targeted and multiagent:
            prompt_file = "env/little_alchemy_2_text/prompts/targeted_multi.txt"
        elif targeted and not multiagent:
            prompt_file = "env/little_alchemy_2_text/prompts/openended_multi.txt"
        elif not targeted and multiagent:
            prompt_file = "env/little_alchemy_2_text/prompts/openeded_multi.txt"
        else:
            prompt_file = "env/little_alchemy_2_text/prompts/openended_single.txt"

        temp = open(prompt_file, 'r').readlines()
        self.intro = " ".join(temp)

    def move(self, state):

        state = self.intro + "\n<bot> RESPONSE:\n" + state

        response = ollama.chat(model='llama3', messages=[
            {
                'role': 'user',
                'content': state,
            },
        ])
        response = response['message']['content']

        return response
