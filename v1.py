import cv2
import numpy as np
from mesures import RACINE, charger, tableau

def compter(img, noyau=25, seuil=40, aire_min=15, aire_max=3000):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (3, 3), 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (noyau, noyau))
    sombre = cv2.morphologyEx(gris, cv2.MORPH_BLACKHAT, k)      # tache sombre locale
    masque = (sombre > seuil).astype(np.uint8)                  # on garde les plus marquées
    masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(masque, connectivity=8)
    boites = [stats[i] for i in range(1, n)
              if aire_min <= stats[i, cv2.CC_STAT_AREA] <= aire_max]
    return len(boites), boites                                  # 1 tache = 1 insecte

def predire(ligne):
    img = cv2.imread(str(RACINE / ligne.split / "images" / ligne.fichier))
    return compter(img)[0]

if __name__ == "__main__":
    df = charger()
    reglage = df[df.groupe == "reglage"].copy()                 # SEULEMENT le réglage
    reglage["pred"] = [predire(l) for l in reglage.itertuples()]
    print(tableau(reglage, "pred"))