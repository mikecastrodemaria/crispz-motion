# crispz-motion

Des vidéos courtes, jusqu'en 4K, à partir de quelques images clés produites par la
famille crispz. FILM (Frame Interpolation for Large Motion, Google Research, ECCV 2022)
calcule toutes les images intermédiaires. Aucun modèle vidéo : les clés viennent des
moteurs d'image crispz par le protocole CLI v1 de la famille (crispz-moss par défaut),
FILM fait le mouvement, ESRGAN passe les images en 4K, ffmpeg écrit le fichier.

[English version](README.md)

> **État : v1.** La page web, l'import des réglages A1111 Animator, les projets, les
> clés importées ou produites par la famille (génération, chaîne img2img, édition,
> inpaint, clé intermédiaire assistée, régénération), le garde-fou de distance, FILM, le
> calage des images, la 4K, les exports, la ligne de commande et le lanceur Pinokio
> fonctionnent.

## Installer, lancer, mettre à jour

**Avec Pinokio** : installe
[crispz-motion.pinokio](https://github.com/mikecastrodemaria/crispz-motion.pinokio)
(Install, Start, Update et Reset en un clic ; les projets sont rangés à côté de l'app et
survivent à un Reset). Sinon, les scripts de ce dossier :

| Windows (cmd) | Windows (PowerShell) | Linux / macOS |
|---|---|---|
| `install.bat` | `install.ps1` | `./install.sh` |
| `start.bat` | `start.ps1` | `./start.sh` |
| `update.bat` | `update.ps1` | `./update.sh` |

- **install** crée deux venvs : `.venv` pour l'app (Pillow seul) et `film/.venv` pour
  FILM et ESRGAN (torch cu128, spandrel). Il télécharge les poids de FILM (138 Mo) et
  écrit `config.json` à partir de `config-sample.json`. Python 3.10 à 3.13.
- **start** installe au premier lancement. Quand GitHub a des commits nouveaux, il
  demande `Update first? [y/N]` ; sans réponse au bout de 10 s, il démarre sans mettre à
  jour. Sans argument (un double-clic), il ouvre la page web ; avec des arguments, il les
  passe à `cm_cli.py` (`start.bat status` affiche l'état).
- **update** lance `git pull --ff-only` : git refuse d'écraser un fichier que tu as
  modifié. Il ne réinstalle que les dépendances dont le fichier a changé, et vérifie que
  torch reste une version CUDA. Il ne touche jamais à `config.json` et liste les clés
  nouvelles du modèle. `--force-deps` (`-ForceDeps` en PowerShell) réinstalle tout.
- Si PowerShell bloque les scripts : `powershell -ExecutionPolicy Bypass -File start.ps1`.
- ffmpeg et ffprobe doivent être dans le PATH (`winget install Gyan.FFmpeg`,
  `sudo apt install ffmpeg`, `brew install ffmpeg`).
- macOS n'a pas CUDA : FILM et ESRGAN y tournent sur le processeur, lentement.
- `python cm_setup.py check` vérifie une installation faite autrement : la version de
  torch et le GPU dans `film/.venv`, les poids de FILM, `config.json`, ffmpeg.

**Deux réglages à poser dans `config.json` :**

- `engines.moss.dir` : le dossier de crispz-moss (`../crispz-moss` par défaut), pour que
  crispz-motion lance le moteur et l'arrête avant FILM. Inutile si tu lances crispz-moss
  toi-même, ou si tu travailles avec tes propres images.
- le modèle ESRGAN pour la 4K (`4x-ClearRealityV1.safetensors` par défaut) : cherché dans
  `esrgan_dir`, sinon dans le dossier ESRGAN de crispz-moss. Sans lui, les images
  au-dessus de la taille des clés passent par Lanczos (plus douces), et le journal du
  rendu le dit.

## La page web

`start.bat` (ou `start.ps1`, `./start.sh`) sans argument lance `cm_server.py` et ouvre
http://127.0.0.1:7870/ dans le navigateur. Ctrl+C dans sa fenêtre l'arrête, ainsi que le
moteur que crispz-motion a lancé.

- **Projets** : ouvrir, dupliquer, supprimer (le dossier part dans `projects/.trash`,
  d'où tu peux le récupérer), créer, importer des réglages Animator.
- **Clés** : la bande des clés dans l'ordre du temps, chacune avec sa source, son prompt,
  sa seed, son temps et son denoise. Glisse une clé pour réordonner (les temps restent,
  les images bougent). Entre deux clés : le score du garde-fou de distance et un bouton
  `+ mid`. Un clic sur une clé l'ouvre : temps, images figées, courbe de vitesse, recette,
  Régénérer (même seed, seed donnée ou nouvelle, avec ou sans les clés construites
  dessus), Supprimer, et la clé suivante faite à partir d'elle : chaîne img2img (une ou
  plusieurs clés ; le curseur de denoise affiche les steps réellement joués), édition
  (avec la note de licence), inpaint (la zone peinte sur l'image, avec un bord adouci).
  Une clé faite depuis une clé du milieu de la bande se place à mi-chemin de la suivante.
- **Ajouter des clés** : dépose des images, ou génère-en une depuis un prompt.
- **Sortie** : résolution, cadence, boucle, segment de fermeture, une clé toutes les N
  secondes ou une durée totale (Re-space), formats, codec, CRF, upscaler, ordre de la 4K,
  moteur.
- **Rendu** : la progression par étape (clés, plan, FILM, calage, taille, export), un
  aperçu, le lecteur vidéo, les fichiers, les temps du dernier rendu, le journal, Annuler.

Toute action qui demande le moteur ou le GPU est un travail : un seul à la fois, affiché
dans la barre du bas avec la dernière ligne du journal. Annuler arrête FILM et ESRGAN
aussitôt (leur worker est tué), et la fabrication des clés entre deux appels au moteur
(l'appel en cours se termine d'abord). Tant qu'un travail tourne sur un projet, la page
ne peut pas modifier ce projet.

## Import des réglages A1111 Animator

L'extension Animator diffusait chaque image : l'image i était l'image i-1 repassée en
img2img avec un denoise faible, la seed + 1, et un prompt qui glissait par poids
(`A :w AND B :1-w`). crispz-motion garde l'idée sur des clés espacées : l'import
transforme `settings.txt` (et `keyframes.csv` s'il existe) en un plan de clés, une par
seconde par défaut.

| Animator | crispz-motion |
|---|---|
| width, height | le ratio des clés, faites vers 2,1 Mpx (1440 x 1440 pour 512 x 512) |
| total_time, fps | des clés à 0, 1, 2… total_time s ; la cadence de sortie |
| image 0 | clé 0, générée (ou l'image initiale si elle est sur le disque) |
| image i, img2img depuis i-1 | clé n, img2img depuis la clé n-1 (`chain`) |
| seed + 1 par image | la seed de l'image de la clé (2763806460, 2763806485…) |
| denoise 0,3 par image | 0,625 par clé (`animator.denoise`) ; le denoise d'une ligne le déplace en proportion, entre 0,25 et 0,75 |
| `A :w AND B :1-w` | la partie de plus fort poids à l'image de la clé |
| `<lora:...>`, embeddings négatifs SD 1.5 | retirés |
| cfg_scale | non repris (LoRA Turbo à CFG 1, où le prompt négatif est sans effet) |
| décalage x / y, zoom, rotation | hors v1 : ignorés |
| vid_mp4 / webm / gif | les formats de sortie |

Chaque écart est écrit dans les notes du plan. Le plan est un tableau : la seed, le
denoise et le prompt de chaque clé prévue se modifient (ou un prompt pour toutes) avant
**Make the planned keys**. Les clés sont faites dans l'ordre et enregistrées une à une :
une fabrication annulée ou en échec reprend là où elle s'est arrêtée, et une clé prévue
dont l'image a été supprimée est refaite.

## Ligne de commande

```bat
start.bat quick a.png b.png --seg 2 --res 4k --loop pingpong
start.bat new smoke --seg 1
start.bat add smoke k0.png k1.png k2.png
start.bat gen smoke --prompt "a pencil drawing of a man with a beard" --seed 2763806460
start.bat render smoke --res 1080p --fps 24 --loop closed --formats mp4,webm,gif
start.bat info smoke
start.bat engine start|stop|status
```

`PROJECT` est un dossier, un nom de dossier sous `projects/`, ou un nom de projet (le plus
récent gagne). Les options de `render` sont enregistrées dans le projet.

### Clés produites par la famille

```bat
start.bat gen smoke --prompt "a pencil drawing of a man ..." --seed 2763806460
start.bat chain smoke --count 10 --denoise 0.625
start.bat chain smoke --count 3 --prompts prompts.txt
start.bat edit smoke "turn his head to the left"
start.bat inpaint smoke --mask mouth.png --prompt "open mouth, smoke coming out"
start.bat guard smoke
start.bat mid smoke k003 k004
start.bat regen smoke k002 --seed 12 --cascade
start.bat remove smoke k005
```

| Commande | Opération du moteur | Ce qu'elle produit |
|---|---|---|
| `gen` | gen | une clé depuis un prompt (LoRA Turbo, 4 steps, CFG 1) |
| `chain` | upscale, facteur 1 | la « boucle creuse » : chaque clé nouvelle est la précédente repassée en img2img à `--denoise`, seed + 1. `--prompts` donne un prompt par clé nouvelle (le dernier se répète) |
| `edit` | edit | la clé suivante par instruction (LoRA Moss-Image-Edit, 50 steps, CFG 4) |
| `inpaint` | inpaint | la clé suivante avec une zone masquée redessinée (blanc = redessiner) |
| `mid` | FILM + upscale | la clé intermédiaire assistée : le milieu FILM de deux clés, nettoyé en img2img à denoise faible (0,2), placé à mi-chemin |
| `regen` | celle de la clé | la clé refaite depuis sa recette et l'image actuelle de sa clé source ; `--cascade` refait ensuite les clés construites dessus |
| `guard` | aucune | la distance entre clés consécutives (voir plus bas) |

Le denoise d'une clé img2img est une fraction des steps que le moteur joue, il avance donc
par paliers : à 4 steps, 0,35 et 0,50 jouent tous deux 2 steps et donnent la même image.
Les clés `chain` et `mid` tournent donc à 8 steps (`chain.steps` dans `config.json`) ; le
journal donne la valeur effective quand elle diffère de celle demandée.

**Ce qui fait bouger le sujet** (mesuré au lot 2, sur le prompt de l'exemple historique
1) : le denoise de la chaîne fait vivre l'image (de 0,625 à 0,75 le personnage tient, à
0,875 il change) ; un prompt nouveau seul bouge à peine la pose à ces valeurs ; l'édition
tourne une tête ou ouvre une bouche, pour environ 20 fois le coût d'une clé en chaîne.
Nomme le style dans l'instruction (« turn his head to the left, keep the black and white
pencil drawing style »), sinon l'édition peut le changer.

**Licence des clés d'édition.** La LoRA Moss-Image-Edit n'indique aucune licence pour ses
poids. Une clé d'édition le dit dans le journal et dans sa recette (`"client": false`,
affiché par `info`) : hors livrable client.

**Garde-fou de distance.** Avant FILM, le rendu note chaque paire de clés consécutives
(écart moyen de vignettes grises de 64 px, de 0 à 255). Au-dessus de `guard.warn` (12) il
avertit, au-dessus de `guard.risk` (25) il conseille une clé intermédiaire : sur un grand
écart, FILM remplit avec des fantômes au lieu d'un mouvement. Le garde-fou ne bloque
jamais un rendu.

**Ordre de l'étape 4K.** `--order film_first` (par défaut) fait tourner FILM à la taille
des clés, puis ESRGAN sur chaque image. `--order keys_first` fait agrandir les clés par le
moteur (ESRGAN puis un léger raffinage Turbo) et fait tourner FILM à la taille de sortie :
clés plus nettes, images intermédiaires plus douces.

Un moteur lancé par crispz-motion reste en marche entre deux commandes (une suite de
commandes ne recharge pas le modèle), et un rendu l'arrête avant FILM. `start.bat engine
stop` l'arrête à la main ; un moteur que tu as lancé toi-même n'est jamais touché.

| Option | Valeurs |
|---|---|
| `--res` | `1mpx`, `1080p` (= `2mpx`), `4k`, `work` (la taille des clés), `LxH`. L'image garde le ratio des clés et tient dans 1280 x 720, 1920 x 1080 ou 3840 x 2160 tourné comme les clés : des clés carrées donnent 720, 1080 ou 2160 de côté |
| `--fps` | 24, 25, 30, 60 ou tout entier |
| `--loop` | `once`, `pingpong`, `closed` (ajoute un segment de la dernière clé vers la première, `--close` secondes) |
| `--seg` | réespace les clés toutes les SEG secondes |
| `--formats` | `mp4`, `webm`, `gif` (liste séparée par des virgules) |
| `--codec`, `--crf` | `x264` ou `x265`, CRF 17 par défaut |
| `--upscaler` | `esrgan` (par défaut au-dessus de la taille des clés) ou `lanczos` |

Une clé a aussi un nombre d'images figées et une courbe de vitesse pour le segment
qu'elle ouvre (`linear`, `ease-in`, `ease-out`, `ease-in-out`) :
`add PROJECT img.png --hold 6 --ease ease-in-out`.

## Le déroulé d'un rendu

1. **Clés** : 2 images ou plus, toutes à la taille de travail (celle de la première clé,
   au plus 2,1 Mpx). Une clé d'un autre ratio est recadrée au centre, et le journal le
   dit. Une clé générée garde sa recette (prompt, seed, la spec envoyée au moteur).
2. **FILM** calcule des milieux récursifs entre chaque paire, en fp16. Le réseau ne répond
   bien qu'à t = 0,5 : il calcule donc 2^N + 1 images par segment, au moins deux par image
   de sortie.
3. **Calage** : chaque image de sortie tombe à son instant exact. Quand cet instant tombe
   entre deux images FILM, les deux sont fondues en proportion. Prendre l'image FILM la
   plus proche faisait saccader le mouvement (pas de 1, 1, 2). Une clé n'apparaît qu'une
   fois : aucune image doublée sur une clé, et l'aller-retour comme la boucle fermée se
   raccordent sans répétition.
4. **Taille** : au-dessus de la taille de travail, ESRGAN (4x-ClearRealityV1) tourne sur
   chaque image, puis un redimensionnement GPU à la taille exacte. Aucune image ne passe
   par la diffusion, donc rien ne scintille. En dessous, Lanczos.
5. **Export** : ffmpeg en mp4 (x264 ou x265, CRF 17, yuv420p), webm (VP9), gif (720 px).
   Le projet reçoit `makemp4.bat` / `.sh` et les autres, pour refaire un fichier depuis
   `frames/` avec d'autres réglages.

Tout est mis en cache par contenu. Un segment FILM porte le hash de ses deux clés et de
ses réglages : changer une clé ne refait que ses deux segments, et changer seulement la
taille de sortie ne refait qu'ESRGAN.

Le GPU est partagé : un moteur lancé par crispz-motion est arrêté avant FILM. Un moteur
que tu as lancé toi-même n'est jamais touché.

## Projets

```
projects/<date>-<nom>/
  project.json         clés, recettes, réglages de sortie, dernier rendu
  keys/  raw/          les clés à la taille de travail, ce que le moteur a rendu
  film/  blend/  final/   caches : images FILM, images fondues, images à la taille de sortie
  frames/              les images de sortie dans l'ordre (liens physiques)
  video.mp4  video.webm  video.gif  make*.bat  make*.sh  render.log
```

## Fichiers

| Fichier | Rôle |
|---|---|
| `cm_server.py` | le serveur de la page web : API JSON, travaux, fichiers des projets |
| `assets/index.html` | la page web (un fichier, sans framework) |
| `cm_animator.py` | import des réglages A1111 Animator, le plan de clés |
| `cm_cli.py` | la ligne de commande |
| `cm_render.py` | un rendu : plan, FILM, calage, taille, contrôles, export |
| `cm_timeline.py` | le calage : instants des images, courbes, images figées, boucles, tailles de sortie |
| `cm_project.py` | un projet sur le disque, ses clés et ses recettes |
| `cm_keys.py` | les clés : import, gen, chaîne, édition, inpaint, milieu, régénération |
| `cm_guard.py` | garde-fou de distance entre clés consécutives |
| `cm_engines.py` | client du protocole CLI v1 et lanceur des apps de la famille |
| `cm_film.py` | pilote les deux workers GPU, cache FILM, recherche du modèle ESRGAN |
| `cm_film_worker.py` | FILM (portage PyTorch), tourne dans `film/.venv` |
| `cm_upscale_worker.py` | ESRGAN sur chaque image, tourne dans `film/.venv` |
| `cm_video.py` | exports ffmpeg et scripts make |
| `cm_setup.py` | installation, mise à jour, vérification des mises à jour, vérification de l'installation (`check`) |
| `cm_core.py` | config, journal, JSON atomique |
| `tools/check_video.py` | contrôle une vidéo : images noires, répétitions, sauts, régularité des pas |
| `tools/bench_*.py` | mesures des lots 0 et 2 |

Tests (processeur, sans GPU) : `film\.venv\Scripts\python -m pytest tests`.

## Licences

- Poids et modèle FILM : Apache 2.0 (google-research/frame-interpolation) ; portage
  PyTorch et export TorchScript : Apache 2.0 (dajes/frame-interpolation-pytorch).
- Moteur d'image par défaut : crispz-moss (MOSS-Image, Apache 2.0) avec la LoRA Turbo
  4 steps (akhaliq/MOSS-Image-SFT-4-steps, Apache 2.0).
- 4x-ClearRealityV1 : voir sa propre licence ; tout modèle ESRGAN lu par spandrel se règle
  dans `config.json`.
