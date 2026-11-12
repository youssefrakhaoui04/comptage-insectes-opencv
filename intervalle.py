import numpy as np
import pandas as pd
from mesures import charger

df = charger()
constante = int(df[df.groupe == "reglage"].n_insectes.median())
p = pd.read_csv("predictions_evaluation.csv", dtype={"jour": str})
err_v4 = (p.pred - p.n_insectes).abs().to_numpy()
err_ref = (constante - p.n_insectes).abs().to_numpy()
print(f"Erreur moyenne : v4 = {err_v4.mean():.2f} | référence (toujours {constante}) = {err_ref.mean():.2f}")

rng = np.random.default_rng(0)
gains = []
for _ in range(2000):
    idx = rng.integers(0, len(p), len(p))
    gains.append(err_ref[idx].mean() - err_v4[idx].mean())
bas, haut = np.percentile(gains, [2.5, 97.5])
print(f"Gain sur la référence : {err_ref.mean() - err_v4.mean():.2f} "
      f"(intervalle à 95 % : de {bas:.2f} à {haut:.2f})")

par_jour = p.assign(erreur_v4=err_v4, erreur_ref=err_ref).groupby("jour")[["erreur_v4", "erreur_ref"]].mean()
print(par_jour.round(2))