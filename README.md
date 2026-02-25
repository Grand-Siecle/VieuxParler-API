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
  }
}
```

## Configuration

Toutes les options sont configurables via variables d'environnement :

| Variable            | Défaut                    | Description                                      |
|---------------------|---------------------------|--------------------------------------------------|
| `MODEL_DIR`         | `./freem_lstm_fairseq`    | Chemin vers le dossier du modèle                 |
| `DEVICE`            | `auto`                    | `auto`, `cpu` ou `cuda`                          |
| `BEAM_SIZE`         | `5`                       | Taille du beam search                            |
| `MAX_TOKENS`        | `200`                     | Nombre max de tokens par inférence               |
| `HOST`              | `0.0.0.0`                 | Adresse d'écoute                                 |
| `PORT`              | `8000`                    | Port d'écoute                                    |
| `WORKERS`           | `1`                       | Nombre de workers Uvicorn                        |
| `LOG_LEVEL`         | `info`                    | Niveau de log (`debug`, `info`, `warning`, etc.) |
| `TRANSLATE_TIMEOUT` | `30`                      | Timeout traduction unique (secondes)             |
| `BATCH_TIMEOUT`     | `300`                     | Timeout traduction batch (secondes)              |

## Tests

```bash
pytest tests/
```

Les tests API (`test_api.py`) fonctionnent sans le modèle — les tests de traduction sont automatiquement ignorés si les fichiers du modèle sont absents. Les tests modèle (`test_model.py`) nécessitent les fichiers du modèle.
