from pathlib import Path
import pandas as pd

RACINE = Path(r"C:\...
JOURS_REGLAGE = ["20221006", "20221012", "20221018"]

def charger():
    df = pd.read_csv("inventaire_filtre.csv", dtype={"jour": str})
    df["groupe"] = ["reglage" if j in JOURS_REGLAGE else "evaluation" for j in df.jour]
    return df

def evaluer(d, col):
    err = d[col] - d.n_insectes
    return {"images": len(d), "MAE": err.abs().mean(),
            "biais": err.mean(), "exact_%": 100 * (err == 0).mean()}

def tableau(d, col):
    res = {"TOUS": evaluer(d, col)}
    for jour, g in d.groupby("jour"):
        res[jour] = evaluer(g, col)
    return pd.DataFrame(res).T.round(2)
