import ollama

def find_nth(haystack, needle, n):
    """ Find the nth occurrence of sub-string in string.
     """
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start + len(needle))
        n -= 1
    return start

class LLM:

    def __init__(self, idx, env, targeted, multiagent):
        self.idx = idx
        self.env = env
        self.done = False

        self.setup(targeted, multiagent)

        self.type = "LLM"
        self.env.reset()

    def setup(self, targeted, multiagent):
        if targeted and multiagent:
            prompt_file = "prompts/targeted_multi.txt"
        elif targeted and not multiagent:
            prompt_file = "prompts/openended_multi.txt"
        elif not targeted and multiagent:
            prompt_file = "prompts/openeded_multi.txt"
        else:
            prompt_file = "prompts/openended_single.txt"

        temp = open(prompt_file, 'r').readlines()
        self.intro = " ".join(temp)


    def parse_action(self, response):

        pos = find_nth(response, "Combination: '", 0)
        error = ""
        if pos != (-1):
            pos = pos + len("Combination: '")
            action_str = response[pos:]
            end_comb = find_nth(action_str, "'", 3)
            action_str = action_str[0:(end_comb + 1)]
            pos = action_str.find(' and ')
            item1 = action_str[:(pos - 1)]
            item2 = action_str[(pos + len(' and ') + 1):-1]
            if item1 in self.env.table and item2 in self.env.table:

                action = [int(self.env.table.index(item1)), int(self.env.table.index(item2))]
            else:
                action = None
                error = "Chosen actions are not in the inventory.\n"

        else:
            action = None
            error = "Invalid action, combination tag not found. Try again.\n"
        return action, error


    def move(self, state):

        state = self.intro + "\n<bot> RESPONSE:\n" + state


        repeat = True
        num_repeats = 0
        while repeat:
            response = ollama.chat(model='llama3', messages=[
                {
                    'role': 'user',
                    'content': state,
                },
            ])
            response = response['message']['content']
            action, error = self.parse_action(response)

            if action is not None:
                repeat = False

            if num_repeats > 100:
                repeat = False

        return action