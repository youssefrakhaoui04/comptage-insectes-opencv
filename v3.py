import cv2
import numpy as np
from evaluation import lire
from v2 import compter

def detecter(g, part_fixe=0.25, tolerance=11, **params):
    """Détecte comme v2, puis ignore les endroits où l'on détecte quelque chose
    dans plus de `part_fixe` des images de la journée (objets fixes : bols, vis...)."""
    par_image = []
    carte = np.zeros((416, 416), np.float32)
    for l in g.itertuples():
        _, boites = compter(lire(l), **params)
        par_image.append(boites)
        m = np.zeros((416, 416), np.uint8)
        for b in boites:
            m[int(b[1]):int(b[1] + b[3]), int(b[0]):int(b[0] + b[2])] = 1
        carte += m
    fixe = (carte / len(g) > part_fixe).astype(np.uint8)
    fixe = cv2.dilate(fixe, np.ones((tolerance, tolerance), np.uint8))
    return [[b for b in boites
             if not fixe[int(b[1] + b[3] // 2), int(b[0] + b[2] // 2)]]
            for boites in par_image]