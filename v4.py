import cv2
import numpy as np
from evaluation import lire
from v2 import zone_plateforme

def _boites(masque, aire_min, aire_max, cote_max):
    masque = cv2.morphologyEx(masque, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(masque, connectivity=8)
    return [stats[i] for i in range(1, n)
            if aire_min <= stats[i, cv2.CC_STAT_AREA] <= aire_max
            and max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) <= cote_max]

def detecter(g, taille_bloc=25, seuil_diff=40, seuil_abs=100,
             aire_min=12, aire_max=3000, cote_max=80, tolerance=5):
    """Fond = image médiane d'un bloc d'images consécutives (le décor, sans les insectes qui bougent).
    Un insecte = une tache plus sombre que ce fond ET sombre dans l'absolu, sur la plateforme."""
    lignes = list(g.itertuples())
    ordre = sorted(range(len(lignes)), key=lambda i: lignes[i].fichier)   # ordre dans le temps
    resultat = [None] * len(lignes)
    for indices in np.array_split(ordre, max(1, len(ordre) // taille_bloc)):
        gris = [cv2.GaussianBlur(cv2.cvtColor(lire(lignes[i]), cv2.COLOR_BGR2GRAY), (3, 3), 0)
                for i in indices]
        mediane = np.median(np.stack(gris), axis=0).astype(np.uint8)
        # on compare au point le plus sombre du fond voisin : tolère un petit décalage de la plateforme
        fond = cv2.erode(mediane, np.ones((tolerance, tolerance), np.uint8)).astype(np.int16)
        for i, gr in zip(indices, gris):
            masque = (((fond - gr.astype(np.int16)) > seuil_diff) & (gr < seuil_abs)
                      & (zone_plateforme(gr) > 0)).astype(np.uint8)
            resultat[i] = _boites(masque, aire_min, aire_max, cote_max)
    return resultat