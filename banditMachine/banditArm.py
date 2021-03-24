import numpy as np
import random as rnd

__author__ = "Ariel Barreiros and Richar Sosa"
__status__ = "Development"


class BanditArm:
    """
    La clase BanditArm reproduce el comportamiento de un brazo de una máquina tragaperras. Y almacena el conjunto de
    recompensas que este brazo ha entregado

    """

    def __init__(self, arm_id, mean, standard_deviation):
        """
        Constructor de la clase

        :param arm_id: es un identificador para el brazo
        :param mean: es la media de las recompensas que genera el brazo
        :param standard_deviation: es la desviación estándar de las recompensas con respecto al valor
                                   establecido como media

        """

        # Initialize variables
        self.__arm_id = arm_id
        self.__mean = mean
        self.__standard_deviation = standard_deviation
        self.__reward_history = np.empty((1, 0))

    def calculate_immediate_reward(self, ):
        """
        Calcula la recompensa del brazo cuando se acciona y guarda este valor en un registro

        :return: recompensa para una muestra

        """

        reward = rnd.gauss(self.__mean, self.__standard_deviation)
        self.__reward_history = np.append(self.__reward_history, reward)
        return reward

    def get_reward_history(self):
        """
        Retorna el campo privado 'reward history'
        :return: reward history
        """
        return self.__reward_history

    def get_arm_id(self):
        """
        Retorna el campo privado 'arm_id'
        :return: arm id
        """
        return self.__arm_id
