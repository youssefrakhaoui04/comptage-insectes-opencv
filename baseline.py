from mesures import charger, tableau

df = charger()
reglage = df[df.groupe == "reglage"]
evaluation = df[df.groupe == "evaluation"].copy()

c = int(reglage.n_insectes.median())
evaluation["pred"] = c
print("Valeur constante choisie sur les journées de réglage :", c)
print(tableau(evaluation, "pred"))