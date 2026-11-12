from mesures import RACINE, charger

def boites_vraies(ligne):
    label = RACINE / ligne.split / "labels" / (ligne.fichier.rsplit(".", 1)[0] + ".txt")
    res = []
    for t in label.read_text().splitlines():
        if t.strip():
            _, cx, cy, bw, bh = map(float, t.split()[:5])
            res.append(((cx - bw / 2) * 416, (cy - bh / 2) * 416,
                        (cx + bw / 2) * 416, (cy + bh / 2) * 416))
    return res

def se_touchent(a, b, marge=4):
    """Deux boîtes (x1, y1, x2, y2) se touchent, à `marge` pixels près."""
    return not (a[2] + marge < b[0] or b[2] + marge < a[0] or
                a[3] + marge < b[1] or b[3] + marge < a[1])
