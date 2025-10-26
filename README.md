# Detection-camera-issue
Ce projet, mené dans le cadre de la vidéosurveillance intelligente de Videtics, vise à détecter automatiquement les dégradations des flux vidéo afin d’en garantir la fiabilité. Il contribue à renforcer la performance des outils, la satisfaction client et le positionnement de l’entreprise.

# 🧠 Image Classification Pipeline (PyTorch)

Ce projet implémente un pipeline complet de **classification d’images** en utilisant PyTorch. Il prend en charge des datasets personnalisés, des augmentations de données, un réseau de neurones modulaire et une boucle d'entraînement optimisée pour le GPU.

---

## ✨ Fonctionnalités principales

- 📂 Dataset personnalisé (`CustomImageDataset`, `EvaluationDataset`)
- 🔁 Augmentations sur mesure via `CustomAugmentedLoader`
- 🧱 Réseau modulaire défini dans `Network.py`
- 🎯 Support multi-sorties avec gestion de plusieurs fonctions de perte
- 📊 Visualisation via TensorBoard
- 🐳 Compatible Docker (image optimisée pour GPU NVIDIA)
- 🔄 Export ONNX automatisé en parallèle de l'entraînement (`export_onnx`)
- ⚙️ Export ONNX compatible avec `onnxsim` et `onnxruntime`

---

## ⚙️ Installation (via Docker)

### 1. Cloner le dépôt

```bash
git clone https://github.com/votre-utilisateur/votre-repo.git
cd votre-repo
``` 

### 2. Lancer l'entraînement via `kronos-cli`

```bash
kronos-cli run train --experiment_name FirstTrain --force
```

---

## 🚀 Utilisation

### 🔧 Entraînement (exemple `kronos.yml`)

```yaml
train:
  script:
    - echo "Démarrage de l'entraînement..."
    - python3 /repo/network_train/train.py
  datasets:
    - "data-3classes"
    - "camstate_ok_images"
  volumes:
    - "SegmentAnythingFull"
```

> 💡 Vous pouvez aussi exécuter l'entraînement manuellement depuis le conteneur :
>
> ```bash
> python3 /repo/network_train/train.py --train_path /datasets/camstate_ok_images --val_path /datasets/data-3classes --batch_size 128 --epochs 20 --n_classes 3 --PATH_OBJECT /volumes/SegmentAnythingFull
> ```

---

## 📈 Visualisation avec TensorBoard

```bash
python3 -m tensorboard.main --logdir /kronos_worker/experiments/images-quality-classification/ --bind_all --port 6005
```

---

## 📤 Export ONNX

L'export du modèle au format ONNX est automatisé via `export_onnx` et peut être exécuté avec :

```bash
kronos-cli run export_onnx --experiment_name ConvertModel --force --dependency FirstTrain
```

> Le modèle exporté peut ensuite être simplifié ou évalué avec `onnxsim` ou `onnxruntime`.

---

## 🎛️ Liste des arguments disponibles

| Argument         | Type   | Valeur par défaut                        | Description                                                                 |
|------------------|--------|------------------------------------------|-----------------------------------------------------------------------------|
| `--train_path`   | str    | `/datasets/camstate_ok_images`           | Dossier contenant les images d'entraînement.                                |
| `--val_path`     | str    | `/datasets/data-3classes`                | Dossier contenant les images de validation.                                 |
| `--batch_size`   | int    | `128`                                    | Taille des batchs (ajuster selon la mémoire GPU disponible).                |
| `--epochs`       | int    | `20`                                     | Nombre d’épochs pour l'entraînement.                                        |
| `--lr`           | float  | `0.001`                                  | Taux d’apprentissage.                                                       |
| `--n_classes`    | int    | `3`                                      | Nombre de classes de sortie.                                                |
| `--freq_save`    | int    | `5`                                      | Fréquence (en épochs) de sauvegarde intermédiaire des checkpoints.          |
| `--PATH_OBJECT`  | str    | `/volumes/SegmentAnythingFull`           | Dossier de sauvegarde pour les checkpoints du modèle.                       |

---

## 💻 Commandes utiles

Voici quelques commandes courantes utilisées avec le projet :

```bash
# Lancer un entraînement avec Docker et Kronos
kronos-cli run train --experiment_name FirstTrain --force

# Suivre les logs d’un conteneur en direct
docker container logs kronos_cli_images-quality-classification_FirstTrain -f

# Ouvrir TensorBoard dans le conteneur (port 6005)
python3 -m tensorboard.main --logdir /kronos_worker/experiments/images-quality-classification/ --bind_all --port 6005

# Exporter le modèle au format ONNX (dépendant d’un entraînement précédent)
kronos-cli run export_onnx --experiment_name ConvertModel --force --dependency FirstTrain

# Ouvrir un terminal dans le conteneur d'entraînement
docker exec -it kronos_cli_images-quality-classification_FirstTrain /bin/bash
```
