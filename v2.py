import cv2
import numpy as np
from mesures import RACINE, charger, tableau

def zone_plateforme(gris, seuil_clair=70, marge=9):
    """La plateforme est claire, le dehors est sombre : on garde la plateforme, un peu rognée."""
    _, m = cv2.threshold(gris, seuil_clair, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(gris)
    enveloppe = cv2.convexHull(max(contours, key=cv2.contourArea))
    zone = np.zeros_like(gris)
    cv2.fillConvexPoly(zone, enveloppe, 1)
    return cv2.erode(zone, np.ones((marge, marge), np.uint8))

def compter(img, noyau=25, seuil=40, seuil_abs=80, aire_min=15, aire_max=3000, cote_max=80):
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gris = cv2.GaussianBlur(gris, (3, 3), 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (noyau, noyau))
    sombre = cv2.morphologyEx(gris, cv2.MORPH_BLACKHAT, k)       # plus sombre que le voisinage
    masque = ((sombre > seuil) & (gris < seuil_abs)              # ET sombre dans l'absolu
              & (zone_plateforme(gris) > 0)).astype(np.uint8)    # ET sur la plateforme
    masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(masque, connectivity=8)
    boites = [stats[i] for i in range(1, n)
              if aire_min <= stats[i, cv2.CC_STAT_AREA] <= aire_max
              and max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) <= cote_max]
    return len(boites), boites

def predire(ligne):
    img = cv2.imread(str(RACINE / ligne.split / "images" / ligne.fichier))
    return compter(img)[0]

if __name__ == "__main__":
    df = charger()
    reglage = df[df.groupe == "reglage"].copy()
    reglage["pred"] = [predire(l) for l in reglage.itertuples()]
    print(tableau(reglage, "pred"))