import numpy as np
from q_learning import q_learning_policies as lp


class NArmedMachine:

    def __init__(self, arms, iterations=1):
        self.arms = arms
        self.iterations = iterations
        self.q_values_evolution = np.empty((len(arms), 0))

    def find_best_arm(self, epsilon_trade_off=0.1):

        q_values = np.zeros(len(self.arms))
        policy = lp.QLearningPolicies()
        next_state_reward_vector = np.zeros(len(self.arms))

        # Repeat the Q learning for the amount of planned iterations
        for i in range(self.iterations):

            # Select an arm by epsilon greedy policy
            selected_arm = policy.epsilon_greedy(epsilon_trade_off, q_values)

            for k in range(len(self.arms)):
                next_state_reward_vector[k] = self.arms[k].get_arm_reward_vector()[i+1]

            q_values[selected_arm] = q_values[selected_arm] + 0.1*(next_state_reward_vector[selected_arm] - q_values[selected_arm])

            self.q_values_evolution = np.append(self.q_values_evolution, np.atleast_2d(q_values).T, axis=1)

    def get_arm_q_values_evolution(self, arm):
        return self.q_values_evolution[arm]
