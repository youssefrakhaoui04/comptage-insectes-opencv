import cv2
import numpy as np
import pandas as pd
from mesures import RACINE, charger
from appariement import boites_vraies
import v4

PARAMS = dict(seuil_diff=40, aire_min=12, taille_bloc=12)   # les mêmes que l'évaluation finale
evaluation = charger().query("groupe == 'evaluation'")

lignes = []
for jour, g in evaluation.groupby("jour"):
    for l, boites in zip(g.itertuples(), v4.detecter(g, **PARAMS)):
        lignes.append((len(boites) - l.n_insectes, l, boites))

def dessiner(ligne, boites, erreur):
    img = cv2.imread(str(RACINE / ligne.split / "images" / ligne.fichier))
    for x1, y1, x2, y2 in boites_vraies(ligne):                      # vérité : rouge, épais
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
    for b in boites:                                                  # détections : cyan, fin
        cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[0] + b[2]), int(b[1] + b[3])), (255, 255, 0), 1)
    cv2.putText(img, f"{ligne.jour} vrai={ligne.n_insectes} detecte={ligne.n_insectes + erreur}",
                (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img

trop = sorted(lignes, key=lambda t: -t[0])[:6]      # les 6 pires sur-comptages
pas_assez = sorted(lignes, key=lambda t: t[0])[:6]  # les 6 pires sous-comptages
vignettes = [dessiner(l, b, e) for e, l, b in trop + pas_assez]
montage = np.vstack([np.hstack(vignettes[i:i + 4]) for i in range(0, 12, 4)])
cv2.imwrite("pires_erreurs.jpg", montage)
print("Les 6 premières vignettes : trop de détections ; les 6 suivantes : pas assez.")