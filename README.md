# 🌿 LeafCheck – Plant Leaf Disease Detection Using Transfer Learning

![Python](https://img.shields.io/badge/Python-3.9--3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12--2.19-orange)
![Flask](https://img.shields.io/badge/Flask-2.3+-lightgrey)
![Dataset](https://img.shields.io/badge/Dataset-PlantVillage-green)

A deep learning mini project that classifies a leaf photo into one of **38 PlantVillage crop-condition classes**
using **transfer learning** (MobileNetV2 and ResNet50), explains each prediction with **Grad-CAM**, and serves
everything through a **Flask web application**.

> **B.Tech – Artificial Intelligence and Data Science**, KGiSL Institute of Technology
> Full project report: [`DL_Project_Report.pdf`](DL_Project_Report.pdf)

## Team

| Roll No. | Name |
|---|---|
| 711724UAD102 | Aaysha asmitha H |
| 711724UAD103 | Abarna A R |
| 711724UAD105 | Akilan C K |
| 711724UAD119 | Bhaavana K |
| 711724UAD125 | Dharshini S |

---

## Features

- **Two transfer-learning backbones** – MobileNetV2 (lightweight, edge-friendly) and ResNet50 (deeper, higher capacity), both pretrained on ImageNet.
- **Two-phase training** – train a new softmax head with the backbone frozen, then fine-tune the top backbone layers at a much lower learning rate.
- **Leakage-aware data split** – stratified 70/15/15 split in which an original photo and its augmented copies always stay in the same set.
- **Full evaluation** – accuracy, top-3 accuracy, macro precision/recall/F1, confusion matrix, per-class F1, model size, inference latency.
- **Grad-CAM** heat-maps to show which leaf regions influenced the prediction.
- **Web interface** – upload a leaf (JPG/PNG/BMP/WEBP), choose a model, see the predicted class, confidence, top-5 probabilities, symptom notes, general advice and a Grad-CAM overlay.
- **Low-confidence warning** – the UI asks for a better photo when confidence is below 60 %.
- **CLI** for data preparation, training and single-image prediction.
- **Unit tests** that run without TensorFlow or the dataset.

## Project structure

```
plant-disease-detection/
├── main.py              # CLI: prepare | train | predict
├── app.py               # Flask website (python app.py)
├── config.py            # all settings in one place
├── requirements.txt
├── src/
│   ├── splitting.py     # stratified, group-aware 70/15/15 split
│   ├── data.py          # tf.data pipelines, augmentation, class weights
│   ├── models.py        # MobileNetV2 / ResNet50 + new softmax head, fine-tuning helper
│   ├── train.py         # two-phase training with early stopping
│   ├── evaluate.py      # metrics, confusion matrix, latency
│   ├── visualize.py     # training curves, confusion matrix, per-class F1, comparison
│   ├── gradcam.py       # Grad-CAM heat-maps
│   ├── predictor.py     # loads saved models and predicts one image
│   └── class_info.py    # readable names + general advice for each class
├── templates/ static/   # website (HTML, CSS, JS)
├── tests/               # pytest suite
├── docs/                # project report (PDF)
├── data/  models/  results/   # created / filled while running the pipeline
```

## Pipeline overview

```
PlantVillage images → split (train/val/test) → preprocess + augment → MobileNetV2 / ResNet50
→ phase 1: train head → phase 2: fine-tune top layers → evaluate → save model
→ Flask app → prediction + top-5 + Grad-CAM
```

**Model:** ImageNet backbone → `GlobalAveragePooling2D` → `Dropout(0.3)` → `Dense(38, softmax)`
**Input:** 224 × 224 RGB, with the backbone-specific `preprocess_input`.

## Getting started

### 1. Install

Python 3.9 – 3.12 is recommended.

```bash
git clone https://github.com/<your-username>/plant-disease-detection.git
cd plant-disease-detection

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

The first training run downloads the ImageNet weights, so an internet connection is needed.

### 2. Get the dataset

Download **PlantVillage** (~54,000 images, 38 classes, use the *color* version), e.g. from Kaggle
(`abdallahalidev/plantvillage-dataset`):

```bash
pip install kaggle        # requires your Kaggle API token
kaggle datasets download -d abdallahalidev/plantvillage-dataset -p data/raw --unzip
```

Or download manually and unzip into `data/raw/`. The script locates the class folders automatically.

### 3. Split the data

```bash
python main.py prepare --raw data/raw          # add --link to save disk space
```

Creates `data/split/{train,val,test}` and `split_manifest.csv`.

### 4. Train and evaluate

```bash
python main.py train                                  # both models
python main.py train --models mobilenetv2             # one model only
python main.py train --head-epochs 3 --ft-epochs 3    # quick trial run
```

A GPU is strongly recommended (see the Colab section); CPU training takes hours.

**Outputs**

- `models/<model>.keras`, `models/class_names.json`
- `results/comparison.csv`, `results/model_comparison.png`
- `results/<model>/` → `metrics.json`, `classification_report.csv`, `confusion_matrix.csv/.png`, `training_curves.png`, `per_class_f1.png`, `history.json`

### 5. Run the web app

```bash
python app.py
```

Open <http://127.0.0.1:5000>. The **Models** page shows the comparison table and plots from `results/`.
Without trained models the app shows a "train first" message.

### 6. Predict from the command line

```bash
python main.py predict --image leaf.jpg --model mobilenetv2
```

### 7. Run the tests

```bash
python -m pytest -q
```

## Training on Google Colab (free GPU)

1. Runtime → Change runtime type → **GPU**.
2. Upload the project zip and the dataset zip to Google Drive, then:

```python
from google.colab import drive; drive.mount('/content/drive')
!unzip -q /content/drive/MyDrive/plant-disease-detection.zip -d /content
!unzip -q /content/drive/MyDrive/plantvillage-dataset.zip -d /content/plant-disease-detection/data/raw
%cd /content/plant-disease-detection
!pip install -q flask
!python main.py prepare --raw data/raw --link
!python main.py train
```

3. Download `models/` and `results/` and copy them into your local project, then run `python app.py`.

## Configuration

All settings are in [`config.py`](config.py):

| Setting | Default |
|---|---|
| Image size / batch size | 224 × 224 / 32 |
| Split (train / val / test) | 70 / 15 / 15 |
| Dropout | 0.3 |
| Phase 1 (frozen backbone) | up to 8 epochs, LR 1e-3 |
| Phase 2 (fine-tune) | up to 12 epochs, LR 1e-5 |
| Unfrozen layers | MobileNetV2: 40, ResNet50: 30 |
| Early-stopping patience | 4 (on `val_loss`) |
| Class weights | on (PlantVillage is imbalanced) |
| Confidence threshold (UI) | 0.60 |
| Max upload size | 8 MB |
| Seed | 42 |

## Results

The figures below are the values **shown in the web interface and transcribed in the project report**
(Chapter 5–6). They were **not re-derived from training logs in the report**. After you run
`python main.py train`, your own reproducible numbers are written to `results/comparison.csv` – use those
as the source of truth.

| Model | Parameters | Test accuracy | Latency |
|---|---|---|---|
| MobileNetV2 | 3.53 M | 97.42 % | 41.8 ms |
| ResNet50 | 25.6 M | 98.68 % | 89.4 ms |
| Custom 4-block CNN (baseline, shown in the UI) | 4.21 M | 92.15 % | 34.2 ms |

**Takeaway:** ResNet50 is the most accurate, MobileNetV2 is about 2× faster with ~7× fewer parameters,
which illustrates the accuracy-versus-efficiency trade-off. Latency depends on hardware and batch size.

## Limitations

- PlantVillage images are clean, single leaves on plain backgrounds; accuracy on real **field photos** is usually much lower.
- Some photos show the same physical leaf from different angles and cannot be grouped automatically, so test accuracy may be somewhat optimistic.
- The model only knows the 38 trained classes and does not check whether the image is a leaf; unfamiliar images may be forced into a known class.
- Softmax confidence is **not** a calibrated probability of a correct diagnosis.
- Grad-CAM shows influential regions; it does not prove a prediction is correct.
- Advice texts are general information, not agronomic prescriptions. Consult a qualified agricultural expert before applying treatments.

## Future work

- Evaluate on real farm images and add an out-of-distribution / "not a leaf" check.
- Convert MobileNetV2 to TensorFlow Lite and benchmark on mobile / edge devices.
- Add image-quality checks (blur, low light, multiple leaves).
- Expand to more crops, disease stages and regions; validate advice with agronomists.
- Downloadable prediction reports (image, class, model, timestamp).

## References

1. Hughes, D. P., & Salathé, M. (2015). *An open access repository of images on plant health to enable the development of mobile disease diagnostics.* arXiv:1511.08060.
2. Sandler, M. et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR.
3. He, K. et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
4. Selvaraju, R. R. et al. (2017). *Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization.* ICCV.
