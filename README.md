# pytorch-alpha-zero

This repository is an implementation of the 2017 DeepMind AlphaZero paper. 

To Do:
1. Implement Tic Tac Toe environment (at least sketch out the API)
2. Write the state class and the MC tree search algorithm.
3. Try running this on Tic Tac Toe
4. Try something like Othello, or mini chess

Big ideas:
1. RL is the only kind of ML that actually lets you achieve super-human performance. Anything else is 
not really learning, it's learning to imitate. And yes, that's something that humans do a lot. But we need
to imitate, and then we need to improve by practicing

1. learn basic concepts by imitation, including basic concepts (supervised ML and some kind of program synthesis, i.e. learning useful abstraction)
2. learn by self-play, trial and error (RL), build up new abstractions that go further than prior techniques, or apply techniques in new ways.# pytorch-alphazero
