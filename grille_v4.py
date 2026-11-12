import itertools
import pandas as pd
from mesures import charger
from evaluation import evaluer_methode
import v4

reglage = charger().query("groupe == 'reglage'")
resultats = []
for seuil_diff, aire_min, taille_bloc in itertools.product([30, 40, 55], [8, 12], [15, 25, 40]):
    tab = evaluer_methode(reglage, lambda g: v4.detecter(g, seuil_diff=seuil_diff,
                                                         aire_min=aire_min, taille_bloc=taille_bloc))
    t = tab.loc["TOUS"]
    resultats.append({"seuil_diff": seuil_diff, "aire_min": aire_min, "taille_bloc": taille_bloc,
                      "MAE": float(t["MAE"]), "biais": float(t["biais"]), "exact_%": float(t["exact_%"]),
                      "rappel_%": float(t["rappel_%"]), "precision_%": float(t["precision_%"])})
    print(resultats[-1])
pd.DataFrame(resultats).to_csv("grille_v4.csv", index=False)