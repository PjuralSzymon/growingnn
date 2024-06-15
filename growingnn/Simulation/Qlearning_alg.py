import time
import random
from ..action import *
from ..structure import *

class QLearningAgent:
    def __init__(self, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.3):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob
        self.q_table = {}  # Słownik przechowujący wartości Q dla każdej pary (stan, akcja)

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, new_value):
        self.q_table[(state, action)] = new_value

    def choose_action(self, actions, state):
        if random.uniform(0, 1) < self.exploration_prob:
            return random.choice(actions)
        else:
            q_values = [self.get_q_value(state, action) for action in actions]
            max_q = max(q_values)
            best_actions = [action for action, q_value in zip(actions, q_values) if q_value == max_q]
            return random.choice(best_actions)

# Globalny agent Q-learninga
global_agent = QLearningAgent()

async def get_action(M, max_time_for_dec, epochs, X_train, Y_train, simulation_score):
    size_of_changes = len(Action.generate_all_actions(M))
    if size_of_changes == 0:
        print("Error")
        return None, 0

    deadline = time.time() + max_time_for_dec
    deepth = 0
    rollouts = 0

    while time.time() < deadline or rollouts <= size_of_changes:
        # Rozpoczęcie nowej symulacji - agent nie zna stanu ani akcji
        state = M.get_state_representation() if M else None
        actions = Action.generate_all_actions(M)

        # Wybór akcji przez agenta
        action = global_agent.choose_action(actions, state)

        # Wykonanie akcji na kopii obiektu M, jeśli istnieje
        new_M = M.deepcopy() if M else None
        if new_M:
            action.execute(new_M)
            #new_M.apply_action(action)

        # Uzyskanie nagrody za wykonanie akcji
        reward = simulation_score.scoreFun(new_M, epochs, X_train, Y_train) if new_M else 0

        # Aktualizacja wartości Q
        if state and action:
            next_state = new_M.get_state_representation()
            q_value = global_agent.get_q_value(state, action)
            next_max_q_value = max(global_agent.get_q_value(next_state, next_action) for next_action in actions)
            new_q_value = q_value + global_agent.learning_rate * (reward + global_agent.discount_factor * next_max_q_value - q_value)
            global_agent.update_q_value(state, action, new_q_value)

        rollouts += 1

    if time.time() > deadline:
        print("More time was needed to analyze all possibilities at least once")

    # Ostateczne wybranie najlepszej akcji po zakończeniu symulacji
    best_action = global_agent.choose_action(actions, state)

    return best_action, deepth, rollouts

# def scoreFun(M, epochs, X_train, Y_train, simulation_score):
#     acc, history = M.gradient_descent(X_train, Y_train, epochs, LearningRateScheduler(LearningRateScheduler.CONSTANT, 0.1) , True)
#     return simulation_score.grade(acc, history)
#     #return max(1.e-17, max_loss - history.get_last('loss'))
