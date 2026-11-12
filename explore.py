from pathlib import Path
import pandas as pd

RACINE = Path(r"C:\Users\Youssef\Documents\insect-counting\data"
              r"\Insect_Detect_detection.v4-insect_detect_416_1class.yolov5pytorch")

lignes = []
for split in ["train", "valid", "test"]:
    for img in (RACINE / split / "images").glob("*.jpg"):
        label = RACINE / split / "labels" / (img.stem + ".txt")
        texte = label.read_text().strip() if label.exists() else ""
        n_insectes = len([l for l in texte.splitlines() if l.strip()])
        lignes.append({
            "split": split,
            "fichier": img.name,
            "jour": img.name[:8],            # ex. 20220905
            "base": img.name.split(".rf.")[0],  # nom sans le hash Roboflow
            "label_existe": label.exists(),
            "n_insectes": n_insectes,
        })

df = pd.DataFrame(lignes)
df.to_csv("inventaire.csv", index=False)

print("Nombre d'images :", len(df))
print("Images sans fichier label :", (~df.label_existe).sum())
print("Noms en double (même photo plusieurs fois) :", df.base.duplicated().sum())
print()
print("Images par jour et par ensemble :")
print(df.pivot_table(index="jour", columns="split", values="fichier",
                     aggfunc="count", fill_value=0, margins=True))
print()
print("Répartition du nombre d'insectes par image :")
print(df.n_insectes.value_counts().sort_index())



# --- Étape 2 : garder les journées d'au moins 50 images ---
par_jour = df.groupby("jour").size()
jours_gardes = par_jour[par_jour >= 50].index
df_filtre = df[df.jour.isin(jours_gardes)].copy()
df_filtre.to_csv("inventaire_filtre.csv", index=False)

print()
print("Journées gardées :", len(jours_gardes), "| images gardées :", len(df_filtre))
print(df_filtre.groupby("jour").n_insectes.agg(["count", "mean"]).round(2))



# --- Aperçu : images avec leurs boîtes d'annotation ---
import cv2
import numpy as np

def dessiner(ligne):
    chemin = RACINE / ligne.split / "images" / ligne.fichier
    img = cv2.imread(str(chemin))
    h, w = img.shape[:2]
    label = RACINE / ligne.split / "labels" / (chemin.stem + ".txt")
    for l in label.read_text().splitlines():
        if not l.strip():
            continue
        _, cx, cy, bw, bh = map(float, l.split()[:5])
        x1, y1 = int((cx - bw / 2) * w), int((cy - bh / 2) * h)
        x2, y2 = int((cx + bw / 2) * w), int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.putText(img, f"{ligne.jour} n={ligne.n_insectes}", (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img

echantillon = df_filtre.sample(12, random_state=0)
images = [dessiner(l) for l in echantillon.itertuples()]
lignes_img = [np.hstack(images[i:i + 4]) for i in range(0, 12, 4)]
cv2.imwrite("apercu.jpg", np.vstack(lignes_img))
print("Aperçu enregistré dans apercu.jpg")




# --- Zoom sur les annotations (les plus petites + au hasard) ---
import random
boites = []
for l in df_filtre.itertuples():
    label = RACINE / l.split / "labels" / (Path(l.fichier).stem + ".txt")
    for ligne in label.read_text().splitlines():
        if ligne.strip():
            _, cx, cy, bw, bh = map(float, ligne.split()[:5])
            boites.append((bw * 416 * bh * 416, l.split, l.fichier, cx, cy, bw, bh))
boites.sort()
aires = pd.Series([b[0] for b in boites])
print("Aire d'une boîte en pixels (image 416x416) :")
print(aires.describe(percentiles=[.1, .5, .9]).round(0))

def recadrer(b, taille=160):
    _, split, fichier, cx, cy, bw, bh = b
    img = cv2.imread(str(RACINE / split / "images" / fichier))
    demi = int(max(bw, bh) * 416 / 2) + 15
    x, y = int(cx * 416), int(cy * 416)
    x1, y1 = int((cx - bw / 2) * 416), int((cy - bh / 2) * 416)
    x2, y2 = int((cx + bw / 2) * 416), int((cy + bh / 2) * 416)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
    crop = img[max(0, y - demi):y + demi, max(0, x - demi):x + demi]
    return cv2.resize(crop, (taille, taille))

choix = boites[:8] + random.Random(0).sample(boites[8:], 8)
crops = [recadrer(b) for b in choix]
montage = np.vstack([np.hstack(crops[i:i + 4]) for i in range(0, 16, 4)])
cv2.imwrite("zoom_boites.jpg", montage)
print("Zoom enregistré dans zoom_boites.jpg")