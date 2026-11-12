import pandas as pd
from mesures import charger, tableau
from evaluation import evaluer_methode
import v4

PARAMS = dict(seuil_diff=40, aire_min=12, taille_bloc=12)   # GELÉS : ne plus les changer

df = charger()
reglage = df[df.groupe == "reglage"]
evaluation = df[df.groupe == "evaluation"].copy()

# 1. Référence : la valeur constante choisie sur les journées de réglage
constante = int(reglage.n_insectes.median())
evaluation["pred_ref"] = constante
reference = tableau(evaluation, "pred_ref")

# 2. La méthode, lancée une seule fois (on garde les comptages image par image)
comptages = {}
def detecter_et_garder(g):
    boites = v4.detecter(g, **PARAMS)
    for l, b in zip(g.itertuples(), boites):
        comptages[l.fichier] = len(b)
    return boites

resultat = evaluer_methode(evaluation, detecter_et_garder)
evaluation["pred"] = evaluation.fichier.map(comptages)

print(f"Référence : toujours {constante}")
print(reference)
print()
print("Soustraction de fond (v4), paramètres :", PARAMS)
print(resultat)

resultat.to_csv("resultats_finaux.csv")
evaluation[["split", "fichier", "jour", "n_insectes", "pred"]].to_csv("predictions_evaluation.csv", index=False)