import cv2
import pandas as pd
from mesures import RACINE
from appariement import boites_vraies, se_touchent

def lire(ligne):
    return cv2.imread(str(RACINE / ligne.split / "images" / ligne.fichier))

def _ligne(nom, c):
    return {"jour": nom, "images": c["images"],
            "MAE": c["abs"] / c["images"], "biais": c["signe"] / c["images"],
            "exact_%": 100 * c["exact"] / c["images"],
            "vrais": c["vrais"], "detections": c["det"],
            "rappel_%": 100 * c["trouves"] / c["vrais"],
            "precision_%": 100 * c["ok"] / max(c["det"], 1)}

def evaluer_methode(df, detecter):
    """detecter(g) -> une liste par image de g, contenant les boîtes (x, y, w, h, ...) détectées."""
    cles = ["images", "abs", "signe", "exact", "vrais", "det", "trouves", "ok"]
    total = dict.fromkeys(cles, 0)
    lignes = []
    for jour, g in df.groupby("jour"):
        c = dict.fromkeys(cles, 0)
        for l, boites in zip(g.itertuples(), detecter(g)):
            det = [(b[0], b[1], b[0] + b[2], b[1] + b[3]) for b in boites]
            vr = boites_vraies(l)
            err = len(det) - l.n_insectes
            c["images"] += 1
            c["abs"] += abs(err)
            c["signe"] += err
            c["exact"] += int(err == 0)
            c["vrais"] += len(vr)
            c["det"] += len(det)
            c["trouves"] += sum(any(se_touchent(v, d) for d in det) for v in vr)
            c["ok"] += sum(any(se_touchent(d, v) for v in vr) for d in det)
        lignes.append(_ligne(jour, c))
        for k in cles:
            total[k] += c[k]
    lignes.append(_ligne("TOUS", total))
    return pd.DataFrame(lignes).set_index("jour").round(1)