from PIL import Image
import matplotlib.pyplot as plt

import numpy as np
import math
import scipy.signal
from pylab import *
import cv2
from scipy.signal import convolve2d
import scipy.stats as st

image_house = np.array(Image.open("images/house2.jpg"),dtype='int32')
image_rectangle = np.array(Image.open("images/img2.jpg").convert('L'),dtype='int32')
image_carrelage = np.array(Image.open("images/carrelage_wikipedia.jpg"),dtype='int32')
image_jussieu = np.array(Image.open("images/Jussieu_wikipedia.jpg"),dtype='int32')


def affichage_14(affichages, titres=None):
    # list[Array|Image]*list[str] -> NoneType
    # effectue entre 1 et 4 affichages avec leurs titres, pour des images ou courbes

    # paramètres :
    #  - liste des affichages (entre 1 et 4)
    #  - liste des titres (entre 1 et 4, autant que de affichages) Optionnelle

    if not type(affichages) == type([]):
        affichages = [affichages]

    if titres is None:
        titres = ['', ] * len(affichages)

    if not type(titres) == type([]):
        titres = [titres]

    nb_affichages = len(affichages)
    if nb_affichages > 4 or nb_affichages < 1:
        raise ValueError('affichage_14 nécéssite 1 à 4 entrées en paramètre')

    if nb_affichages != len(titres):
        raise ValueError('affichage_14 nécéssite autant de titres que d\'affichages')

    courbes = False
    for i in range(0, nb_affichages):
        s = plt.subplot(101 + 10 * nb_affichages + i)
        s.set_title(titres[i])
        if len(affichages[i].shape) == 2 and affichages[i].shape[0] > 1 and affichages[i].shape[1] > 1:
            # on affiche une image
            s.imshow(affichages[i], cmap="gray", interpolation='nearest', aspect='equal')
        else:
            # il s'agit d'une seule ligne, à afficher comme une courbe
            plt.plot(affichages[i])
            courbes = True

    agrandissement_h = nb_affichages
    agrandissement_v = nb_affichages * 2 if courbes else nb_affichages
    params = plt.gcf()
    plSize = params.get_size_inches()
    params.set_size_inches((plSize[0] * agrandissement_v, plSize[1] * agrandissement_h))
    plt.show()


def module_affichage(module):
    # permet de transformer un module de DFT en une version jolie à afficher
    module = np.array(module, dtype='float32')
    ind_max = np.where(module == np.max(module.flatten()))
    module[ind_max] = 0.0
    module[ind_max] = np.max(module.flatten())
    module = sqrt(module)
    return sqrt(module)




def gradient(image):
    """ Array -> tuple[Array*Array]"""
    sobelx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
    sobely = sobelx.T

    dx = convolve2d(image, sobelx, "same")
    dy = convolve2d(image, sobely, "same")

    return (dx, dy)





def noyau_gaussien(sigma):
    """ float -> Array """

    x = np.linspace(-3 * sigma, 3 * sigma, int(6 * sigma + 2))
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)

    return kern2d / kern2d.sum()


def harris(image, sigma, kappa):
    """ Array*float*float->Array """
    # calcule des gradients
    Ix, Iy = gradient(image)

    gauss = noyau_gaussien(sigma)

    # calcule de A
    A11 = convolve2d(Ix * Ix, gauss, "same")
    A22 = convolve2d(Iy * Iy, gauss, "same")
    A12 = convolve2d(Ix * Iy, gauss, "same")

    # calcule de R
    det = (A11 * A22) - (A12 * A12)
    trace = A11 + A22

    R = det - kappa * trace ** 2

    return R


def maxlocal(image_harris, seuil):
    """ Array*float -> Array """

    coin = np.zeros(image_harris.shape)

    for i in range(2, image_harris.shape[0] - 2):
        for j in range(2, image_harris.shape[1] - 2):

            if (np.amax(image_harris[i - 1:i + 2, j - 1:j + 2]) == image_harris[i, j]) and image_harris[i, j] > seuil:
                coin[i, j] = image_harris[i, j]
                image_harris[i - 1:i + 2, j - 1:j + 2] = 0

    return np.array(coin)


def maxlocal_fast(image_harris, seuil):
    """Array*float -> Array"""

    # thresholding

    filtre = np.array([[-1, -1, -1],
                       [-1, +8, -1],
                       [-1, -1, -1]])

    coin = convolve2d(image_harris, filtre, "same")

    # non maximum supression
    #     F = OFilter(4, 3)
    #     B = F.ordfilt2(coin)
    coin[image_harris < seuil] = 0
    coin[coin > 0] = image_harris[coin > 0]

    return coin


def coord_maxlocal(image_extrema, seuil):
    """ Array*float -> list[list[int,int]] """

    print("max extrema ", np.amax(image_extrema))

    indecies = np.dstack(
        np.unravel_index(np.argsort(image_extrema.ravel()), (image_extrema.shape[0], image_extrema.shape[1])))
    indecies = indecies.squeeze()

    print("before", indecies.shape[0])

    for i in range(len(indecies)):
        if (image_extrema[indecies[i][0], indecies[i][1]] > 0):
            print("i", i)
            indecies = indecies[i:]
            break

    print("after", indecies.shape[0])
    truc = int(seuil * len(indecies) / 100)

    print("pourcentage", truc, len(indecies))

    return indecies[-truc:]


# def test_harris(image, pourcent):
#     R = harris(image, 1, 0.06)
#     coin = maxlocal(R, 20000000)
#     # print(len(coin))
#     ind = coord_maxlocal(coin, pourcent)
#     ind = np.array(ind)
#     # print(ind[-3:])
#     plt.figure(figsize=(10, 10))
#
#     #     plt.subplot(121)
#
#     plt.imshow(image, cmap="gray")
#
#     plt.axis('tight')
#     plt.axis('off')
#
#     plt.scatter(ind[:, 1], ind[:, 0], color="red", marker="o")
#
#     plt.show()
#

if __name__ == "__main__":
    from time import sleep

    webcam = cv2.VideoCapture(0)

    sleep(2)

    while True:
        check, frame = webcam.read()
        
        image_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        R = harris(image_gray, 1, 0.06)
        coin = maxlocal(R, 20000000)
        # print(len(coin))
        ind = coord_maxlocal(coin, 20)
        ind = np.array(ind)

        for i in range(ind.shape[0]):
            print(i)
            cv2.circle(frame, (ind[i, 1], ind[i, 0]), 5, (0, 255, 0), -1)

        cv2.imshow("Capturing", image_house)

    webcam.release()

    print("Camera off.")

    print("Program ended.")

    cv2.destroyAllWindows()