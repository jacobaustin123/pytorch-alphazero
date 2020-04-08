import pyspiel
import numpy as np

game = pyspiel.load_game('othello')
state = game.new_initial_state()
print(state)
print("terminal:", state.is_terminal())
print("current player:", state.current_player())
print(state.legal_actions())

while not state.is_terminal():
    moves = state.legal_actions()
    state.apply_action(np.random.choice(moves))
    print(moves)
    print(state.current_player())
    print(state)

print(state.is_terminal())
print(state.rewards())
print(sum(state.observation_tensor()[64:128]))
print(sum(state.observation_tensor()[128:]))
