import numpy as np


def get_data_pair_for_plotting(data, precision):
    """
    Esta función toma un vector numpy, lo organiza de menor a mayor y le redondea los valores. Además determina que
    cantidad de veces está repetido cada valor, con el objetivo de crear 2 vectores que puedan ser representados
    gráficamente

    :param data: un numpy vector
    :param precision: las posciones decimales que se esperan en el resultado
    :return: un par de vectores (x,y). El primero 'x' es un vector organizado que contiene una única vez los valores
             que se encontraban dentro del juego de datos. El segundo 'y' es un vector que contiene la cantidad de
             ocurrencias que tiene, dentro del juego de datos, cada uno de los valores del vector 'x'
    """

    np.sort(data)
    for i in range(len(data)):
        data[i] = data[i].__round__(precision)

    (x, y) = np.unique(data, return_counts=True)
    return np.array(x), np.array(y)
