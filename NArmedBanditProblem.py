import numpy as np
import matplotlib.pyplot as plt
from banditMachine.banditArm import BanditArm
from banditMachine import nArmedMachine as bM


def main():

    # Creación de los brazos
    cantidad_de_brazos = 10
    arms = []
    for i in range(cantidad_de_brazos):
        arms.append(BanditArm("Arm_" + str(i), (i + np.random.uniform(0, 1)), np.random.uniform(0, 1)))

    # Adición de los brazos a la máquina
    bandit_machine = bM.NArmedMachine(arms)

    # Órden de aprendizaje
    iteraciones = 3000
    bandit_machine.find_best_arm(iteraciones, 0.1)

    # Gráfica del comportamiento de los q_values de cada brazo
    plt.figure(1)
    legend_plot_1 = []
    for i in range(len(arms)):
        data = bandit_machine.get_arm_q_values_evolution(i)
        legend_plot_1.append(bandit_machine.get_arm(i).get_arm_id())
        plt.plot(data)
    plt.legend(legend_plot_1)
    plt.show()

    # Gráfica de las recompensas entregadas por cada brazo en las oportunidades que fueron seleccionados
    plt.figure(2)
    legend_plot_2 = []
    for i in range(len(arms)):
        data = bandit_machine.get_arm(i).get_reward_history()
        legend_plot_2.append(bandit_machine.get_arm(i).get_arm_id())
        plt.plot(data)
    plt.legend(legend_plot_2)
    plt.show()


if __name__ == "__main__":
    main()
