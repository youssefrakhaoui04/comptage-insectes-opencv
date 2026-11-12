import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mesures import charger
from evaluation import evaluer_methode
import v4

reglage = charger().query("groupe == 'reglage'")
reference = (reglage.n_insectes - 1).abs().mean()      # erreur de "toujours 1" sur ces journées

lignes = []
for tb in [6, 8, 10, 12, 15, 20, 25, 40]:
    t = evaluer_methode(reglage, lambda g: v4.detecter(g, seuil_diff=40, aire_min=12,
                                                       taille_bloc=tb)).loc["TOUS"]
    lignes.append({"taille_bloc": tb, "MAE": float(t["MAE"]), "exact_%": float(t["exact_%"]),
                   "rappel_%": float(t["rappel_%"]), "precision_%": float(t["precision_%"])})
    print(lignes[-1])
res = pd.DataFrame(lignes)
res.to_csv("courbe_bloc.csv", index=False)

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
a.plot(res.taille_bloc, res.MAE, "o-", label="soustraction de fond")
a.axhline(reference, color="gray", linestyle="--", label="toujours 1")
a.set_xlabel("images par bloc"); a.set_ylabel("erreur moyenne (insectes par image)"); a.legend()
b.plot(res.taille_bloc, res["rappel_%"], "o-", label="rappel")
b.plot(res.taille_bloc, res["precision_%"], "s-", label="précision")
b.set_xlabel("images par bloc"); b.set_ylabel("%"); b.legend()
fig.suptitle("Journées de réglage")
fig.tight_layout(); fig.savefig("courbe_bloc.png", dpi=150)
print("Figure enregistrée : courbe_bloc.png")