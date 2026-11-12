import numpy as np
import pandas as pd

p = pd.read_csv("predictions_evaluation.csv", dtype={"jour": str})
lignes = []
for jour, g in p.groupby("jour"):
    g = g.sort_values("fichier").reset_index(drop=True)          # ordre dans le temps
    for k, idx in enumerate(np.array_split(range(len(g)), max(1, len(g) // 12))):
        b = g.iloc[idx]
        erreur = b.pred - b.n_insectes
        lignes.append({"jour": jour, "bloc": k, "images": len(b),
                       "de": b.fichier.iloc[0][9:17], "a": b.fichier.iloc[-1][9:17],
                       "erreur_moy": erreur.abs().mean(), "biais": erreur.mean()})
print(pd.DataFrame(lignes).round(2).to_string(index=False))