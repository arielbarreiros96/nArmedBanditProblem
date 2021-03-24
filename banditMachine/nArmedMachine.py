import numpy as np
from q_learning import q_learning_policies as lp


class NArmedMachine:
    """
    Esta clase define una máquina tragaperras con 'arms' cantidad de brazos, y 'tries' cantidad de recompensas
    entregadas por cada brazo

    """

    def __init__(self, arms):
        """
        Constructor de la clase

        :param arms: son el conjunto de brazos que definen a la máquina

        q_values_evolution: es un vector que acumula el comportamiento de los q_valores asociados al aprendizaje, para
                            cada brazo, durante el proceso de encontrar el mejor brazo de la máquina

        """
        self.__arms = arms
        self.__q_values_evolution = np.empty((len(arms), 0))

    def find_best_arm(self, tries=1, epsilon_trade_off=0.1):
        """
        Este método básicamente actualiza los q_valores mediante RL para luego poder determinar cuál es el mejor de los
        brazos de la máquina.
        El comportamiento de los q-values queda almacenado en el vector q_values_evolution, para su posterior análisis

        :param tries: es la cantidad de intentos que va a realizar el aprendizaje con el objetivo de encontrar el mejor
                      brazo de la máquina
        :param epsilon_trade_off: por defecto es 0.1 y define la probabilidad que que el algoritmo seleccione una acción
                                  de exploración el lugar de explotación

        """

        q_values = np.zeros(len(self.__arms))
        policies = lp.QLearningPolicies()

        # Repeat the Q learning for the amount of planned iterations
        for i in range(tries):

            # Select an arm by epsilon greedy policies
            selected_arm = policies.epsilon_greedy(epsilon_trade_off, q_values)
            # Recibe una recompensa de la acción que se seleccionó
            arm_reward = self.__arms[selected_arm].calculate_immediate_reward()
            # Actualiza los q_values en función de la recompensa y vuelve a comenzar
            q_values[selected_arm] = q_values[selected_arm] + 0.1*(arm_reward - q_values[selected_arm])

            # Guarda los q_valores en un arreglo para luego poder analizar el comportamiento
            self.__q_values_evolution = np.append(self.__q_values_evolution, np.atleast_2d(q_values).T, axis=1)

    def get_arm_q_values_evolution(self, arm):
        """
        Retorna el comportamiento de los q_values para el brazo requerido

        :param arm: brazo del cual se desea conocer el comportamiento de los q_values
        :return: comportamiento de los q_values a través del aprendizaje
        """
        return self.__q_values_evolution[arm]

    def get_arm(self, arm):
        """
        :param arm: número del brazo que se necesita
        :return: el brazo que se pide
        """
        return self.__arms[arm]
