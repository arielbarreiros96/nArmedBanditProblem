import numpy as np
import random as rnd

__author__ = "Ariel Barreiros and Richar Sosa"
__status__ = "Development"


class BanditArm:
    """
    La clase BanditArm reproduce el comportamiento de un brazo de una máquina tragaperras. Definida una cantidad
    de muestras, la media de recompensa que entrega la máquina y la desviación estándar de estas recompensas se crea
    un vector de recompensa que comprende las recompensas para cada intento.

    """

    def __init__(self, arm_id, samples_count, mean, standard_deviation):
        """
        Constructor de la clase

        :param arm_id: es un identificador para el brazo
        :param samples_count: es la cantidad de muestras para las que se va a generar recompensa
        :param mean: es la media de las recompensas que genera el brazo
        :param standard_deviation: es la desviación estándar de las recompensas con respecto al valor
                                   establecido como media

        """

        # Initialize variables
        self.arm_id = arm_id
        self.samples_count = samples_count
        self.mean = mean
        self.standard_deviation = standard_deviation
        self.reward_vector = np.zeros(samples_count)
        # Fill reward vector
        self.fill_reward_vector()

    def calculate_immediate_reward(self):
        """
        Calcula la recompensa para una muestra

        :return: recompensa para una muestra

        """
        return rnd.gauss(self.mean, self.standard_deviation)

    def fill_reward_vector(self):
        """
        Rellena el vector de recompensas para cada muestra, de acuerdo a los parámetros definidos

        """
        for i in range(self.samples_count):
            self.reward_vector[i] = self.calculate_immediate_reward()

    def get_arm_reward_vector(self):
        """
        Retorna el vector de recompensas del brazo

        :return: vector de recompensas del brazo

        """
        return self.reward_vector
