import numpy as np
import matplotlib.pyplot as plt
from banditMachine.banditArm import BanditArm
from banditMachine import nArmedMachine as bm

samples = 5000


def main():

    arms = []
    for i in range(5):
        arms.append(BanditArm("Arm_" + str(i), samples, (i + np.random.uniform(0, 1)), np.random.uniform(0, 1)))

    bandit_machine = bm.NArmedMachine(arms, samples-1)
    bandit_machine.find_best_arm(0.1)

    for i in range(len(arms)):
        data = bandit_machine.get_arm_q_values_evolution(i)
        plt.plot(data)
    plt.show()


if __name__ == "__main__":
    main()
