# Compter des insectes sur des photos de piège photographique (OpenCV, sans deep learning)

Exercice proposé par Alexandre Conanec (Agriodor). Objectif : compter les insectes présents sur chaque image d'un jeu de données public (Insect Detect, Zenodo 7725941, images 416x416, annotations YOLO), sans distinguer les espèces, uniquement en traitement d'image classique avec OpenCV, et dire honnêtement ce que vaut la méthode.

**Temps passé : environ 10 heures au total** (exercice reçu le mardi 10 novembre vers 18h). Mardi soir, environ 3 h 30 (de 19h à 22h-23h) : exploration des données et premiers scripts de mesure. Mercredi 11 novembre (jour férié), environ 5 h (de 13h à 18h, avec des pauses) : versions 1 à 4 de la méthode et leurs réglages. Environ 1 h 30 en plus, pour l'évaluation finale, l'analyse des erreurs et la rédaction, terminées le jeudi 12 novembre vers 15h.

**Utilisation d'une IA :** j'ai travaillé avec l'assistant Claude pour discuter de la démarche et écrire une partie du code. J'ai lancé chaque étape moi-même, regardé les images et relu le code. C'est la première fois que je traite un problème de comptage uniquement avec OpenCV : d'habitude, pour ce genre de tâche, j'utiliserais un réseau de neurones convolutif (avec Keras et TensorFlow). J'ai donc utilisé l'IA pour apprendre quels outils d'OpenCV employer (flou, « blackhat », composantes connexes, soustraction de fond) et comment les combiner. Je peux expliquer chaque fichier du dépôt.

## Résultat en bref

Sur 4 journées que je n'avais pas utilisées pour régler la méthode (472 images), l'erreur moyenne est de **0,75 insecte par image**, contre **1,01** pour une méthode « bête » qui répond toujours 1. Le gain est de 0,26 (intervalle à 95 % : de 0,12 à 0,38). La méthode ne gagne pas partout, et elle a des défauts nets, décrits plus bas.

| Journée (2022) | Images | Erreur : toujours 1 | Erreur : ma méthode | Images au compte exact (ma méthode) |
|---|---|---|---|---|
| 7 octobre | 114 | 1,51 | 1,39 | 37 % |
| 17 octobre | 107 | 0,74 | 0,39 | 69 % |
| 19 octobre | 185 | 1,09 | 0,68 | 47 % |
| 20 octobre | 66 | 0,35 | 0,39 | 82 % |
| **Ensemble** | **472** | **1,01** | **0,75** | **54 %** (contre 34 %) |

Sur ces 472 images : rappel 83 % (part des insectes annotés retrouvés), précision 83 % (part de mes détections qui correspondent à un insecte annoté), biais -0,1 (je ne compte ni trop ni trop peu en moyenne).

## Les données et le cadrage

- 1335 images, 23 journées de captation (la date est dans le nom du fichier). Je ne tiens pas compte du découpage train/valid/test de Roboflow : les photos d'une même journée sont mélangées dans les trois, et je n'entraîne rien.
- Comme demandé, je garde les journées d'au moins 50 images : 7 journées (6, 7, 12, 17, 18, 19, 20 octobre 2022), soit **1101 images et 1931 insectes annotés**. Le résultat est le même si on ne compte que dans `train`.
- Les insectes sont petits : la boîte annotée médiane fait environ 18x18 pixels, et 10 % des boîtes font moins de 9x9.
- Les images ont été écrasées de 16:9 en 416x416 : les formes sont déformées.
- Sur les 1101 images, 25 n'ont aucun insecte, 487 en ont un, 382 deux, 163 trois, et 44 quatre ou plus.

## Comment j'ai mesuré

- **Journées de réglage : 6, 12 et 18 octobre** (629 images). Tous les essais et réglages se font là. **Journées d'évaluation : 7, 17, 19 et 20 octobre** (472 images). Je ne les ai utilisées qu'**une seule fois**, à la fin, avec des paramètres figés. J'avais vu quelques images de ces journées dans un aperçu au début, sans rien régler dessus.
- **Référence : « toujours 1 »**, valeur choisie sur les journées de réglage (la médiane). Sur le réglage, elle donne 0,64 d'erreur et environ 52 % d'images exactes. Une méthode qui ne bat pas cela ne sert à rien.
- **Mesures** : erreur absolue moyenne du nombre d'insectes par image, biais (détections moins vérité), part d'images au compte exact, et, pour comprendre les erreurs, rappel et précision. Une détection compte comme bien placée si sa boîte touche une boîte annotée à 4 pixels près : c'est une approximation simple, pas un appariement un pour un.
- **Intervalle d'incertitude** : tirage au hasard d'images (2000 fois). Les images d'une même journée se ressemblent, donc l'incertitude réelle est un peu plus grande.
- **Nombre d'essais** : environ 40 combinaisons de paramètres testées sur le réglage (12 pour la version 3, 18 plus 8 pour la version 4, et les premières versions). Plus on essaie, plus un bon chiffre peut être dû au hasard : c'est une raison de plus pour la séparation réglage/évaluation.

## Ce que j'ai essayé, dans l'ordre (chiffres sur les journées de réglage)

| Version | Idée | Erreur moyenne | Rappel / précision |
|---|---|---|---|
| Référence | toujours 1 | 0,64 | |
| v1 | taches plus sombres que leur voisinage (« blackhat » d'OpenCV), seuil, composantes connexes | 18,6 | |
| v2 | + restriction à la plateforme + il faut être sombre dans l'absolu | 4,2 | 88 % / 25 % |
| v3 | + on ignore les endroits détectés dans plus de 25 % des images de la journée (objets fixes) | 1,0 à 1,1 | 73 à 79 % / 57 à 61 % |
| v4 | soustraction de fond : médiane de blocs d'images consécutives, on garde ce qui est plus sombre que ce fond | **0,4** | 87 % / 90 % |

Ce qui m'a fait avancer, et ce qui m'a trompé :
- **v1** détectait environ 19 fausses alertes par image. J'avais pensé que les pétales bleus en étaient la cause : l'image de diagnostic a montré que non. C'étaient les bords de la plateforme, les bols blancs brillants et leur ombre, les vis métalliques, et les creux entre les lobes des grandes fleurs.
- **v3** supprimait les objets fixes, mais aussi les insectes posés dessus (ils se posent souvent sur les bols) : rappel de 96 % à 72 % le 6 octobre. Elle ne battait pas la référence.
- J'ai regardé les insectes manqués le 18 octobre : au milieu de l'image (pas au bord), très sombres (pas un problème de seuil), mais minuscules (médiane de 52 pixels carrés). Baisser le seuil de taille augmentait le rappel (jusqu'à 83 %) mais baissait la précision à 50 % : l'erreur moyenne n'y gagnait rien.
- Faire varier les seuils (grille de 12 puis de 18 combinaisons) ne changeait presque rien. **Le seul paramètre qui comptait était la taille des blocs** : plateau entre 10 et 15 images (erreur 0,4), puis dégradation (0,9 à 40 images). J'ai retenu 12, au milieu du plateau.

La courbe de l'erreur, du rappel et de la précision selon la taille des blocs (journées de réglage) est produite par `python courbe_bloc.py` (fichiers `courbe_bloc.csv` et `courbe_bloc.png`).

## La méthode retenue (v4)

1. Image en niveaux de gris, léger flou.
2. Les images d'une journée sont triées par heure et découpées en blocs d'environ 12 images consécutives. L'**image médiane du bloc** représente le décor, sans les insectes qui bougent.
3. Un pixel est candidat s'il est plus sombre que le fond (de plus de 40 niveaux de gris, en comparant au point le plus sombre du fond voisin pour tolérer un petit décalage), s'il est sombre dans l'absolu (moins de 100), et s'il se trouve sur la plateforme (zone claire de l'image, légèrement rognée).
4. Composantes connexes ; je garde les taches entre 12 et 3000 pixels carrés, de côté inférieur à 80. **Le nombre de taches est le nombre d'insectes.**

Paramètres figés : `seuil_diff=40`, `aire_min=12`, `taille_bloc=12`.

## Ce qui ne marche pas

`python pires.py` produit `pires_erreurs.jpg` : les 6 pires sur-comptages (6 premières vignettes) et les 6 pires sous-comptages (6 dernières), sur les journées d'évaluation ; vérité en rouge, mes détections en cyan. Le tableau par blocs vient de `erreur_par_bloc.py`.

- **Blocs étalés sur plusieurs heures : trop de détections.** Le bloc du 7 octobre de 11 h 14 à 14 h 07 a un biais de +3,4 insectes par image : le soleil bouge, les ombres des bols aussi, et la médiane ne représente plus le décor. Ce n'est pas systématique (un bloc de 3 heures le 20 octobre est bon).
- **Blocs très serrés dans le temps : insectes manqués.** Le 7 octobre, entre 18 h 01 et 18 h 14, le biais est de -1,4 à -1,8. Un insecte qui reste immobile sur plusieurs photos finit par faire partie du fond et n'est plus compté. C'est ce qui arrive aux insectes posés sur les bols.
- **Conclusion : découper en blocs d'un nombre fixe d'images est le mauvais critère** ; il faudrait des fenêtres de durée fixe. Je ne l'ai pas essayé, pour ne pas réutiliser les journées d'évaluation.
- **Insectes collés** : le 19 octobre, une image avec 6 insectes (plusieurs sur les mêmes fleurs, au contact) est comptée 3. Un groupe de deux insectes devient une seule tache.
- **Insectes très petits** : environ un insecte annoté sur dix mesure moins de 9x9 pixels ; ils se perdent dans le bruit (sur le réglage, le 18 octobre, la version 4 avec des blocs de 25 images ne retrouvait que 79,5 % des insectes).
- **La vérité elle-même est discutable.** Les classes d'origine ont été fusionnées en « insect », y compris, d'après la description Zenodo, « shadow » (l'ombre d'un insecte). Dans mes zooms, au moins une boîte est posée sur une ombre, et un petit point immobile est annoté comme insecte sur plusieurs photos du 7 octobre. Je ne l'ai pas quantifié. Une partie de mes erreurs vient donc peut-être de la vérité, pas de la méthode.
- **La journée du 7 octobre est la plus difficile** : près de 2,4 insectes par image, bols présents, photos groupées en fin de journée. Mon gain sur la référence y est faible (1,39 contre 1,51).
- **Le 20 octobre, la référence est légèrement meilleure** (0,35 contre 0,39), sur 66 images : trop peu pour conclure.
- La méthode a besoin de beaucoup d'images de la même journée pour calculer le fond ; elle ne marcherait pas image par image.

## Ce que je ferais avec plus de temps

1. Fenêtres de **durée** fixe (par exemple 30 minutes) au lieu d'un nombre d'images.
2. Normaliser la lumière de chaque image avant de la comparer au fond.
3. Garder une trace dans le temps des insectes immobiles, pour ne pas les laisser entrer dans le fond.
4. Séparer les insectes collés (par exemple avec une transformée de distance).
5. Compter proprement la part d'ombres dans les annotations.

## Lancer le code

1. Installer les bibliothèques : `pip install opencv-python numpy pandas matplotlib`.
2. Télécharger le jeu de données (Zenodo 7725941, version « v4 insect_detect_416_1class ») et l'extraire dans `data/`, pour obtenir `data/Insect_Detect_detection.v4-insect_detect_416_1class.yolov5pytorch`.
3. `python explore.py` : inventaire des images, filtre à 50 images par journée (écrit `inventaire_filtre.csv`).
4. `python evaluation_finale.py` : évaluation finale sur les 4 journées d'évaluation, paramètres figés (écrit `resultats_finaux.csv` et `predictions_evaluation.csv`). Durée : environ [X] minutes sur mon ordinateur.
5. `python intervalle.py` : erreur à deux décimales et intervalle d'incertitude, à partir de `predictions_evaluation.csv` (il faut donc avoir lancé `evaluation_finale.py` avant).

Les fichiers de résultats et les images ne sont pas dans le dépôt : ils sont créés en lançant les scripts. `explore.py` écrit `inventaire.csv`, `inventaire_filtre.csv`, `apercu.jpg` et `zoom_boites.jpg` ; `evaluation_finale.py` écrit `resultats_finaux.csv` et `predictions_evaluation.csv` ; `courbe_bloc.py` écrit `courbe_bloc.csv` et `courbe_bloc.png` ; `pires.py` écrit `pires_erreurs.jpg`.

`python v2.py` relance la version 2 sur les journées de réglage. Quelques scripts exploratoires ne sont pas dans le dépôt : le diagnostic de la version 1 (images annotées), l'étude des insectes manqués le 18 octobre, la grille de paramètres de la version 3 et les scripts qui lançaient les versions 2 à 4 sur le réglage. Leurs résultats sont décrits plus haut.

| Fichier | Rôle |
|---|---|
| `explore.py` | inventaire, filtre, aperçu des images annotées |
| `mesures.py`, `evaluation.py`, `appariement.py` | séparation réglage/évaluation, métriques |
| `v1.py` à `v4.py` | les quatre versions de la méthode |
| `baseline.py` | méthode de référence « toujours la même valeur » |
| `grille_v4.py`, `courbe_bloc.py` | essais de paramètres de la version 4, courbe de la taille des blocs |
| `pires.py`, `erreur_par_bloc.py` | analyse des erreurs (pires cas, erreur par bloc avec son heure) |
| `intervalle.py` | erreur à deux décimales et intervalle d'incertitude |

## Données

Insect Detect detection dataset, M. Sittinger, Zenodo 7725941, licence CC BY 4.0. Les images ne sont pas incluses dans ce dépôt.
