import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import special
import warnings


def f(t):
    return pow(math.e, -t) * np.sin(np.sqrt(t))


def real_func(w):
    return (-1j * pow(math.e, 1j * w * pow(np.pi, 2) - pow(np.pi, 2) + 1j * np.sqrt(pow(np.pi, 2))) / (4 * (1j * w - 1))) +\
        (1j * pow(math.e, 1j * w * pow(np.pi, 2) - pow(np.pi, 2) - 1j * np.sqrt(pow(np.pi, 2))) / (4 * (1j * w - 1))) +\
        (1j * pow(math.e, -1j * w * pow(np.pi, 2) - pow(np.pi, 2) + 1j * np.sqrt(pow(np.pi, 2))) / (4 * (1j * w + 1))) -\
        (1j * pow(math.e, -1j * w * pow(np.pi, 2) - pow(np.pi, 2) - 1j * np.sqrt(pow(np.pi, 2))) / (4 * (1j * w + 1))) -\
        ((np.sqrt(np.pi) * pow(math.e, (1 / (4 * (1j * w - 1)))) * special.erfi((2 * (1j * w - 1) *
        np.sqrt(pow(np.pi, 2)) + 1j) / 2 * np.sqrt(1j * w - 1))) / (8 * pow((1j * w - 1), 3 / 2))) -\
        ((np.sqrt(np.pi) * pow(math.e, (1 / (4 * (1j * w - 1)))) * special.erfi((2 * (1j * w - 1) *
        np.sqrt(pow(np.pi, 2)) - 1j) / 2 * np.sqrt(1j * w - 1))) / (8 * pow((1j * w - 1), 3 / 2))) +\
        ((np.sqrt(np.pi) * pow(math.e, (-1 / (4 * (1j * w + 1)))) * special.erfi((2 * (-1j * w - 1) *
        np.sqrt(pow(np.pi, 2)) + 1j) / 2 * np.sqrt(-1j * w - 1))) / (8 * np.sqrt(-1j * w - 1) * (1j * w + 1))) +\
        ((np.sqrt(np.pi) * pow(math.e, (-1 / (4 * (1j * w + 1)))) * special.erfi((2 * (-1j * w - 1) *
        np.sqrt(pow(np.pi, 2)) - 1j) / 2 * np.sqrt(-1j * w - 1))) / (8 * np.sqrt(-1j * w - 1) * (1j * w + 1))) - \
        (-(-1j * pow(math.e, 1j * w * 0 - 0 + 1j * np.sqrt(0)) / (4 * (1j * w - 1))) +\
        (1j * pow(math.e, 1j * w * 0 - 0 - 1j * np.sqrt(0)) / (4 * (1j * w - 1))) +\
        (1j * pow(math.e, -1j * w * 0 - 0 + 1j * np.sqrt(0)) / (4 * (1j * w + 1))) -\
        (1j * pow(math.e, -1j * w * 0 - 0 - 1j * np.sqrt(0)) / (4 * (1j * w + 1))) -\
        ((np.sqrt(np.pi) * pow(math.e, (1 / (4 * (1j * w - 1)))) * special.erfi((2 * (1j * w - 1) *
        np.sqrt(0) + 1j) / 2 * np.sqrt(1j * w - 1))) / (8 * pow((1j * w - 1), 3 / 2))) -\
        ((np.sqrt(np.pi) * pow(math.e, (1 / (4 * (1j * w - 1)))) * special.erfi((2 * (1j * w - 1) *
        np.sqrt(0) - 1j) / 2 * np.sqrt(1j * w - 1))) / (8 * pow((1j * w - 1), 3 / 2))) +\
        ((np.sqrt(np.pi) * pow(math.e, (-1 / (4 * (1j * w + 1)))) * special.erfi((2 * (-1j * w - 1) *
        np.sqrt(0) + 1j) / 2 * np.sqrt(-1j * w - 1))) / (8 * np.sqrt(-1j * w - 1) * (1j * w + 1))) +\
        ((np.sqrt(np.pi) * pow(math.e, (-1 / (4 * (1j * w + 1)))) * special.erfi((2 * (-1j * w - 1) *
        np.sqrt(0) - 1j) / 2 * np.sqrt(-1j * w - 1))) / (8 * np.sqrt(-1j * w - 1) * (1j * w + 1))))


def img_func(w):
    return (-pow(math.e, 1j * w * pow(np.pi, 2) - pow(np.pi, 2) + 1j * np.sqrt(pow(np.pi, 2))) / (4 * (1j * w - 1))) +\
        (pow(math.e, 1j * w * pow(np.pi, 2) - pow(np.pi, 2) - 1j * np.sqrt(pow(np.pi, 2))) / (4 * (1j * w - 1))) -\
        (pow(math.e, -1j * w * pow(np.pi, 2) - pow(np.pi, 2) + 1j * np.sqrt(pow(np.pi, 2))) / (4 * (1j * w + 1))) +\
        (pow(math.e, -1j * w * pow(np.pi, 2) - pow(np.pi, 2) - 1j * np.sqrt(pow(np.pi, 2))) / (4 * (1j * w + 1))) +\
        ((np.sqrt(np.pi) * 1j * pow(math.e, (1 / (4 * (1j * w - 1)))) * special.erfi((2 * (1j * w - 1) *
        np.sqrt(pow(np.pi, 2)) + 1j) / 2 * np.sqrt(1j * w - 1))) / (8 * pow((1j * w - 1), 3 / 2))) +\
        ((np.sqrt(np.pi) * 1j * pow(math.e, (1 / (4 * (1j * w - 1)))) * special.erfi((2 * (1j * w - 1) *
        np.sqrt(pow(np.pi, 2)) - 1j) / 2 * np.sqrt(1j * w - 1))) / (8 * pow((1j * w - 1), 3 / 2))) +\
        ((np.sqrt(np.pi) * 1j * pow(math.e, (-1 / (4 * (1j * w + 1)))) * special.erfi((2 * (-1j * w - 1) *
        np.sqrt(pow(np.pi, 2)) + 1j) / 2 * np.sqrt(-1j * w - 1))) / (8 * np.sqrt(-1j * w - 1) * (1j * w + 1))) +\
        ((np.sqrt(np.pi) * 1j * pow(math.e, (-1 / (4 * (1j * w + 1)))) * special.erfi((2 * (-1j * w - 1) *
        np.sqrt(pow(np.pi, 2)) - 1j) / 2 * np.sqrt(-1j * w - 1))) / (8 * np.sqrt(-1j * w - 1) * (1j * w + 1))) - \
        (-(-pow(math.e, 1j * w * 0 - 0 + 1j * np.sqrt(0)) / (4 * (1j * w - 1))) +\
        (pow(math.e, 1j * w * 0 - 0 - 1j * np.sqrt(0)) / (4 * (1j * w - 1))) -\
        (pow(math.e, -1j * w * 0 - 0 + 1j * np.sqrt(0)) / (4 * (1j * w + 1))) +\
        (pow(math.e, -1j * w * 0 - 0 - 1j * np.sqrt(0)) / (4 * (1j * w + 1))) +\
        ((np.sqrt(np.pi) * 1j * pow(math.e, (1 / (4 * (1j * w - 1)))) * special.erfi((2 * (1j * w - 1) *
        np.sqrt(0) + 1j) / 2 * np.sqrt(1j * w - 1))) / (8 * pow((1j * w - 1), 3 / 2))) +\
        ((np.sqrt(np.pi) * 1j * pow(math.e, (1 / (4 * (1j * w - 1)))) * special.erfi((2 * (1j * w - 1) *
        np.sqrt(0) - 1j) / 2 * np.sqrt(1j * w - 1))) / (8 * pow((1j * w - 1), 3 / 2))) +\
        ((np.sqrt(np.pi) * 1j * pow(math.e, (-1 / (4 * (1j * w + 1)))) * special.erfi((2 * (-1j * w - 1) *
        np.sqrt(0) + 1j) / 2 * np.sqrt(-1j * w - 1))) / (8 * np.sqrt(-1j * w - 1) * (1j * w + 1))) +\
        ((np.sqrt(np.pi) * 1j * pow(math.e, (-1 / (4 * (1j * w + 1)))) * special.erfi((2 * (-1j * w - 1) *
        np.sqrt(0) - 1j) / 2 * np.sqrt(-1j * w - 1))) / (8 * np.sqrt(-1j * w - 1) * (1j * w + 1))))


def fourier(f, a, b, n_1, n_2, m_min, m_max):
    t_list = np.linspace(a, b, n_1)
    chi_list = np.linspace(m_min, m_max, n_2)

    f_real = np.zeros(n_2, dtype='complex')
    f_img = np.zeros(n_2, dtype='complex')

    for i, chi in enumerate(chi_list):
        int_real = f(t_list) * np.cos(chi * t_list)
        int_img = f(t_list) * np.sin(chi * t_list)
        f_real[i] = np.trapz(int_real, t_list)
        f_img[i] = np.trapz(int_img, t_list)

    return f_real, f_img


if __name__ == "__main__":
    warnings.simplefilter(action="ignore", category=RuntimeWarning)

    a = 0
    b = pow(np.pi, 2)
    n_1 = 1000
    n_2 = 1000
    m_min = 4
    m_max = 10
    sin_answ = []
    cos_answ = []

    w = np.linspace(m_min, m_max, n_2)
    cos_answ, sin_answ = fourier(f, a, b, n_1, n_2, m_min, m_max)

    w_2 = np.linspace(m_min, 8, n_2)
    real_answ = real_func(w_2)
    img_answ = img_func(w_2)

    plt.plot(w, sin_answ, "r", w, cos_answ, "b")
    plt.grid()
    plt.show()

    plt.plot(w_2, real_answ, "b", w_2, cos_answ, "r")
    plt.grid()
    plt.show()
