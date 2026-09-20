# VieuxParler-API

API REST pour la traduction du français ancien (XVIIe siècle) vers le français contemporain, basée sur un modèle LSTM Fairseq entraîné sur le corpus FreEM.

> **Usage local uniquement.** Cette API n'est pas conçue pour un déploiement en production. Il n'y a ni authentification, ni HTTPS, ni rate-limiting. Elle est destinée à un usage de recherche et de développement en local.

## Prérequis

- Python 3.9+
- PyTorch (`pip install torch` ou via l'index CPU : `pip install torch --index-url https://download.pytorch.org/whl/cpu`)
- Les fichiers du modèle FreEM LSTM dans le dossier `freem_lstm_fairseq/` (non inclus dans le dépôt)

### Structure attendue du modèle

```
freem_lstm_fairseq/
├── model/
│   └── checkpoint_best.pt
├── data_norm_bin_4000/
│   ├── dict.src.txt
│   └── dict.trg.txt
├── bpe_joint_4000.model
└── bpe_joint_4000.vocab
```

## Installation

### CPU uniquement

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Avec support GPU (CUDA)

Si vous disposez d'un GPU NVIDIA avec CUDA :

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Puis configurez la variable d'environnement :

```bash
export DEVICE=cuda   # forcer le GPU
export DEVICE=auto   # GPU si disponible, sinon CPU (défaut)
```

> Si `DEVICE=cuda` est défini mais qu'aucun GPU n'est disponible, l'API refuse de démarrer avec une erreur explicite.

## Lancement

### Directement avec Python

```bash
python main.py
```

L'API démarre sur `http://localhost:8000`. Le modèle se charge au démarrage (peut prendre quelques secondes).

### Avec Docker Compose

#### Version CPU (par défaut)

```bash
docker compose up --build
```

#### Version GPU (NVIDIA CUDA)

Prérequis : [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installé sur l'hôte.

```bash
docker compose --profile gpu up --build api-gpu
```

Pour arrêter la version CPU avant de lancer la version GPU (ou inversement) :

```bash
docker compose down              # arrête le service CPU
docker compose --profile gpu up --build api-gpu
```

> **Attention :** les deux services utilisent le port 8000. Ne lancez jamais les deux en même temps.

Le dossier `freem_lstm_fairseq/` est monté en lecture seule dans le conteneur.

## Endpoints

| Méthode | Route               | Description                              |
|---------|----------------------|------------------------------------------|
| GET     | `/health`           | État de l'API et infos device (CPU/GPU)  |
| GET     | `/metrics`          | Métriques Prometheus                     |
| POST    | `/translate`        | Traduire un texte unique                 |
| POST    | `/translate/batch`  | Traduire une liste de textes             |
| POST    | `/translate/stream` | Traduction en streaming (SSE)            |

La documentation interactive est disponible sur `http://localhost:8000/docs` (Swagger UI).

## Exemples

### Traduire un texte

```bash
curl -X POST http://localhost:8000/translate \
  -H "Content-Type: application/json" \
  -d '{"text": "Que voulez-vous donc?"}'
```

Réponse :

```json
{
  "original": "Que voulez-vous donc?",
  "translated": "Que voulez-vous donc ?",
  "model_version": "freem-lstm-v1"
}
```

### Traduire un lot de textes

```bash
curl -X POST http://localhost:8000/translate/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Il estoit une fois un homme.",
      "Je ne sçay pas.",
      "Elle avoit de beaux yeux."
    ],
    "batch_size": 32
  }'
```

Réponse :

```json
{
  "translations": [
    "Il était une fois un homme.",
    "Je ne sais pas.",
    "Elle avait de beaux yeux."
  ],
  "processing_time": 1.2345,
  "count": 3
}
```

### Traduction en streaming (SSE)

```bash
curl -X POST http://localhost:8000/translate/stream \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Que voulez-vous donc?", "Il estoit une fois."]}'
```

Réponse (flux Server-Sent Events) :

```
data: Que voulez-vous donc ?

data: Il était une fois.

event: done
data: stream complete
```

### Vérifier l'état de l'API

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "ok",
  "model_loaded": true,
  "version": "freem-lstm-v1",
  "device": {
    "device": "cpu",
    "cuda_available": false,
    "device_config": "auto"
  },
  "inference": {
    "token_budget": 4000,
    "token_budget_mode": "auto",
    "oom_cap": null,
    "fp16": false,
    "beam_size": 5,
    "coalesce_requests": true,
    "split_over_tokens": 200
  }
}
```

## Configuration

Toutes les options sont configurables via variables d'environnement :

| Variable               | Défaut                 | Description                                                                 |
|------------------------|------------------------|-----------------------------------------------------------------------------|
| `MODEL_DIR`            | `./freem_lstm_fairseq` | Chemin vers le dossier du modèle                                            |
| `DEVICE`               | `auto`                 | `auto`, `cpu` ou `cuda`                                                     |
| `BEAM_SIZE`            | `5`                    | Largeur du beam search. `1` est ~4× plus rapide mais change quelques sorties |
| `FP16`                 | `false`                | Inférence en demi-précision (GPU seulement). Peut changer des sorties        |
| `MAX_TOKENS`           | `auto`                 | Budget de tokens par lot fairseq. `auto` = calculé depuis la mémoire GPU libre, ou un entier fixe |
| `CPU_MAX_TOKENS`       | `4000`                 | Budget utilisé sur CPU quand `MAX_TOKENS=auto`                              |
| `MAX_TOKENS_FLOOR`     | `1024`                 | Plancher du budget automatique (GPU)                                        |
| `MAX_TOKENS_CEIL`      | `16000`                | Plafond du budget automatique (GPU)                                         |
| `GPU_MEMORY_FRACTION`  | `0.6`                  | Part de la mémoire GPU libre allouée à un lot                               |
| `BYTES_PER_TOKEN`      | `1048576`              | Octets GPU par token source (beam 5, fp32), utilisés pour le budget automatique. À calibrer avec `scripts/parity_bench.py` |
| `SPLIT_OVER_TOKENS`    | `200`                  | Les lignes plus longues (en tokens) sont découpées à la ponctuation. `0` désactive |
| `SPLIT_SEGMENT_TOKENS` | `40`                   | Taille visée des segments après découpage                                   |
| `COALESCE_REQUESTS`    | `true`                 | Fusionner les requêtes `/translate/batch` concurrentes en un seul appel modèle |
| `HOST`                 | `0.0.0.0`              | Adresse d'écoute                                                            |
| `PORT`                 | `8000`                 | Port d'écoute                                                               |
| `WORKERS`              | `1`                    | Nombre de workers Uvicorn                                                   |
| `LOG_LEVEL`            | `info`                 | Niveau de log (`debug`, `info`, `warning`, etc.)                            |
| `TRANSLATE_TIMEOUT`    | `30`                   | Timeout traduction unique (secondes)                                        |
| `BATCH_TIMEOUT`        | `300`                  | Timeout traduction batch (secondes)                                         |

## Traitement par lots

Un *token* désigne ici une pièce SentencePiece (l'unité que voit le modèle), plus le marqueur de fin de phrase que fairseq ajoute. Une ligne de texte courante fait 10 à 60 tokens.

### Ce que fait `/translate/batch`

1. **Vrais lots.** Toute la liste est passée en un seul appel à `model.translate(list)`. Fairseq trie les lignes par longueur, les regroupe en lots dont le nombre total de tokens ne dépasse pas le *budget de tokens* (`MAX_TOKENS`), et restitue les sorties dans l'ordre d'entrée. Auparavant chaque ligne faisait l'objet d'un appel modèle séparé : le « batch » n'était qu'une boucle Python.
2. **Lignes vides** : renvoyées telles quelles (`""`) sans passer par le modèle. **Lignes identiques** (titres courants, etc.) : traduites une seule fois.
3. **Lignes longues.** Au-delà de `SPLIT_OVER_TOKENS` (200 tokens, l'ancienne limite qui faisait planter tout le lot), la ligne est coupée après la ponctuation (`; : , . ! ?`) en segments d'environ `SPLIT_SEGMENT_TOKENS` tokens (repli sur les mots s'il n'y a pas de ponctuation), chaque segment est traduit, puis les traductions sont recollées avec une espace. Sur le jeu de test, le chrF des lignes de plus de 200 tokens passe de 58,5 à 96,9 avec cette coupe.
4. **Budget de tokens adaptatif** (`MAX_TOKENS=auto`). Sur GPU, avant chaque appel : `mémoire libre × GPU_MEMORY_FRACTION / BYTES_PER_TOKEN`, mis à l'échelle selon le beam et le fp16, borné entre `MAX_TOKENS_FLOOR` et `MAX_TOKENS_CEIL`. La même image s'adapte donc à des GPU de tailles différentes. Sur CPU, valeur fixe `CPU_MAX_TOKENS`. Le budget n'est jamais inférieur à la ligne la plus longue de l'appel.
5. **Repli sur out-of-memory.** Si le GPU manque de mémoire, le cache PyTorch est vidé, le budget est divisé par deux et l'appel rejoué ; le plafond ainsi trouvé est mémorisé pour la suite.
6. **Fusion des requêtes concurrentes** (`COALESCE_REQUESTS`). Un thread unique vide la file des requêtes en attente et les traite en un seul appel modèle, puis redistribue les résultats. Si le lot fusionné échoue, chaque requête est rejouée séparément : une requête fautive n'entraîne pas les autres.

### Contrat client

- `POST /translate/batch` renvoie toujours `translations` de même longueur et dans le même ordre que `texts`.
- `batch_size` est conservé pour compatibilité mais ne dimensionne plus rien. `batch_size=1` sélectionne le **chemin de référence** : un appel modèle par ligne, sans découpage ni déduplication ni fusion (l'ancien comportement).
- Sur CPU comme sur GPU, à `BEAM_SIZE=5` en fp32, les sorties du chemin par lots sont identiques à celles du chemin de référence (voir [Résultats mesurés](#résultats-mesurés)). `BEAM_SIZE` et `FP16` peuvent modifier des sorties : ils restent à leur valeur de référence par défaut.

### Observabilité

`GET /health` expose un bloc optionnel `inference` (budget effectif, mode, plafond OOM, fp16, beam). Métriques Prometheus ajoutées : `token_budget`, `oom_fallbacks_total`, `lines_split_total`, `requests_coalesced_total`, `coalesced_batch_requests`.

### Mesurer la parité et le débit

`scripts/parity_bench.py` (présent dans les images Docker) échantillonne le jeu de test FreEMnorm inclus dans le zip du modèle par classe de longueur, traduit l'échantillon par le chemin de référence puis par lots, et affiche : sorties identiques et exemples de différences, débit des deux chemins, budget et pic mémoire GPU, octets par token mesurés (pour calibrer `BYTES_PER_TOKEN`), chrF par classe de longueur.

```bash
docker compose --profile gpu run --rm api-gpu python scripts/parity_bench.py --n 400
FP16=true  docker compose --profile gpu run --rm api-gpu python scripts/parity_bench.py --n 400
BEAM_SIZE=1 docker compose --profile gpu run --rm api-gpu python scripts/parity_bench.py --n 400
```

### Résultats mesurés

Mesures du 20 septembre 2026 avec `parity_bench.py --n 400` : 326 lignes (13 865 tokens) du jeu de test, 80 par classe de longueur plus les 6 lignes de plus de 200 tokens que compte le jeu. GPU : NVIDIA GeForce RTX 4070 Ti (12 Go), torch 2.5.1+cu124 ; CPU : même machine (WSL2). `BEAM_SIZE=5`, fp32, `MAX_TOKENS=auto` sauf mention.

| Device | Chemin de référence (ancien comportement) | Chemin par lots | Accélération |
|--------|-------------------------------------------|-----------------|--------------|
| GPU    | 24,5 s · 13,3 lignes/s · 565 tok/s        | 2,8 s · 115 lignes/s · 4 881 tok/s | ×8,6 |
| CPU    | 88,1 s · 3,7 lignes/s · 157 tok/s         | 30,6 s · 10,7 lignes/s · 453 tok/s | ×2,9 |

- **Parité.** Sur les 320 lignes de 200 tokens ou moins, sorties identiques au chemin de référence, sur GPU comme sur CPU (vérifié aussi contre le code d'origine de `main`, même débit à 0,1 s près). Les 6 lignes de plus de 200 tokens faisaient planter l'ancien code (`AssertionError: Sentences lengths should not exceed max_tokens=200`) ; coupées, leur chrF passe de 58,5 à 96,9.
- **Le chemin de référence est limité par le surcoût par appel**, environ 75 ms par ligne sur GPU quelle que soit la précision : l'ancien code n'allait que 3,6 fois plus vite sur GPU que sur CPU. C'est le regroupement en lots qui exploite le GPU. Une requête `POST /translate` d'une seule phrase ne bénéficie donc pas de l'accélération.
- **Mémoire.** Sur ce GPU le chemin par lots a coûté 75 KiB par token (pic de 487 MiB au-dessus du modèle avec un budget de 6 457 tokens, 1,1 GiB à 16 000), soit 7 fois moins que la valeur par défaut de `BYTES_PER_TOKEN` (1 MiB). Le budget automatique est donc prudent ; l'abaisser n'a pas apporté de gain mesurable sur un lot de cette taille.

Variantes GPU, chemin par lots sur le même échantillon :

| Réglage                                | Temps | Sorties                                              |
|----------------------------------------|-------|------------------------------------------------------|
| `BEAM_SIZE=5`, fp32 (défaut)           | 2,8 s | identiques à la référence                            |
| `BEAM_SIZE=5`, `FP16=true`             | 2,6 s | chrF identique par classe, budget 13 039             |
| `BEAM_SIZE=1`, fp32                    | 1,9 s | 1 ligne sur 320 diffère                              |
| `BYTES_PER_TOKEN=153178` (valeur calibrée) | 2,8 s | identiques, budget 16 000 au lieu de 6 457, sans gain |

## Tests

```bash
pytest tests/
```

Les tests API (`test_api.py`) et les tests du moteur de lots (`test_batching.py`, faux modèle injecté) fonctionnent sans le modèle — les tests de traduction sont automatiquement ignorés si les fichiers du modèle sont absents. Les tests modèle (`test_model.py`) nécessitent les fichiers du modèle.
