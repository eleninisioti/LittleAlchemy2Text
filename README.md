
## What is LittleAlchemy2Text?

This is a text-based version of the game Little Alchemy 2 that can be played both by humans and Large Language Models.
We have implemented it by extending [Wordcraft](https://github.com/minqi/wordcraft), a python-based implementation of Little
Alchemy 2 that enabled playing the game with reinforcement learning agents.


How did we extend Wordcraft?

* added open-ended tasks. These start with the same items that Little Alchemy 2 starts with and have no target item
* added support for multiple agents. The agents are not in the same environment but receive information about the actions of others
* engineered prompts for instructing LLMs to play the task
* some bug fixes (eg ensuring that tasks are set deterministically by the seed, dealing with items missing from the data base)

## How to use

### Installing dependencies

We have tested our code on Linux and Mac with Python version 3.11

To install all necessary package dependencies you can run:

    conda env create -f environment.yml

### Playing with humans and LLMs


To enable the use of LLMs we are using the [ollama](https://github.com/ollama/ollama-python) library.
This is an API to various opensource LLMs, such as LLama 3, Phi 3 Mistral and Gemma 2.

In order to be able to use a model you need to first pull it, e.g.

    ollama pull llama3

We have provided a script for illustrating how humans and LLMs can play a game. Upon running:

    python play.py

you will be asked how many human and LLM players there will be. Human players will need to perform actions through the command line.




## Cite this work

If you use this code in your work, please cite our paper:

    @article{nisioti_2024, 
    title={Collective Innovation in Groups of Large Language Models},
    author={Eleni Nisioti and Sebastian Risi and Ida Momennejad and Pierre-Yves Oudeyer and Clément Moulin-Frier},
    year={2024},
    booktitle = {The 2023 {Conference} on {Artificial} {Life}},
    publisher = {MIT Press},
    }
    




