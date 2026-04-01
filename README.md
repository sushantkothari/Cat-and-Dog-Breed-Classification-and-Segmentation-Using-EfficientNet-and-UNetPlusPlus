# Cat & Dog Breed Classification and Segmentation Using EfficientNet and UNet++

## Overview

This project is a dual-task computer vision system that performs **fine-grained pet breed classification** and **semantic segmentation** on images from the Oxford-IIIT Pet Dataset. It combines the representational power of EfficientNet-B0 for 37-class breed classification with a UNet++ segmentation model backed by an EfficientNet-B2 encoder to produce pixel-accurate pet masks — all wrapped in an interactive Gradio interface.

The pipeline covers the full machine learning lifecycle: dataset preparation, augmentation, transfer learning, backbone fine-tuning, evaluation, and a live inference demo with segmentation overlays and confidence heatmaps.

---

## Key Highlights

- 37-class fine-grained breed classification (cats and dogs)
- EfficientNet-B0 with custom classification head using BatchNorm + Dropout
- Two-stage training: frozen backbone transfer learning followed by full fine-tuning
- UNet++ segmentation with EfficientNet-B2 encoder and SCSE decoder attention
- Combined Tversky + Focal loss for boundary-aware segmentation
- Mixed precision training with AMP for faster GPU utilization
- Cosine annealing LR scheduler with linear warmup for segmentation
- Early stopping and best model checkpointing for both tasks
- Per-class accuracy analysis and confusion matrix visualization
- Gradio app with breed predictions, segmentation overlays, and confidence heatmaps

---

## Results

![Results](Results.png)

---

## System Architecture

### Classification Pipeline

```
Input Image
    ↓
Augmentation (RandomCrop, Flip, Rotation, ColorJitter)
    ↓
EfficientNet-B0 Backbone (Pretrained on ImageNet)
    ↓
Custom Head: BatchNorm1d → Dropout(0.5) → Linear(37 classes)
    ↓
CrossEntropyLoss (label_smoothing=0.1)
    ↓
AdamW + CosineAnnealingLR
    ↓
Phase 1: Frozen Backbone → Phase 2: Full Fine-Tuning
    ↓
Breed Prediction + Top-5 Confidence Scores
```

### Segmentation Pipeline

```
Input Image
    ↓
Augmentation (Flip, Affine, ColorJitter, GridDistortion)
    ↓
UNet++ Encoder: EfficientNet-B2 (ImageNet Weights)
    ↓
Decoder with SCSE Attention Blocks
    ↓
Combined Loss: 0.6 × Tversky + 0.4 × Focal
    ↓
AdamW + LinearLR Warmup (3 ep) → CosineAnnealingLR
    ↓
Binary Mask: Pet (1) vs Background (0)
```

---

## Dataset

This project uses the [Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).

### Dataset Description

- 37 pet breeds (25 dog breeds, 12 cat breeds)
- ~200 images per class
- Per-pixel trimap annotations for segmentation (pet / background / uncertain)
- Images vary significantly in pose, scale, and lighting

### Trimap Annotation Mapping

| Trimap Value | Meaning       | Mapped To |
|:---:|:---:|:---:|
| 1 | Pet foreground | 1 (pet) |
| 2 | Background | 0 (background) |
| 3 | Uncertain boundary | 0 (background) |

### Citation

O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, *Cats and Dogs*, IEEE Conference on Computer Vision and Pattern Recognition, 2012.

---

## Machine Learning Methodology

### Classification

**Data Preparation**

Class labels are extracted from image filenames by stripping the trailing index (e.g., `Abyssinian_1.jpg` → `Abyssinian`). An 80/20 train/val split is applied with stratified-style label coverage.

**Augmentation (Training)**

- Resize to 300×300 then RandomResizedCrop to 256×256
- RandomHorizontalFlip
- RandomRotation (±25°)
- ColorJitter (brightness, contrast, saturation)
- RandomGrayscale (p=0.1)
- ImageNet normalization

**Model**

EfficientNet-B0 with pretrained ImageNet weights. The backbone is frozen in Phase 1 and the custom head is trained. In Phase 2, all parameters are unfrozen and fine-tuned end-to-end at a lower learning rate.

**Custom Classification Head**

```python
nn.Sequential(
    nn.BatchNorm1d(num_features),
    nn.Dropout(0.5),
    nn.Linear(num_features, 37)
)
```

**Loss & Optimizer**

- CrossEntropyLoss with `label_smoothing=0.1`
- AdamW with weight decay `1e-4`
- CosineAnnealingLR (`T_max=10`)
- Mixed precision with `torch.amp`

---

### Segmentation

**Data Preparation**

All images with corresponding `.png` trimap annotations are used. An 80/20 random split is applied across the full dataset.

**Augmentation (Training)**

- Resize to 256×256
- HorizontalFlip (p=0.5)
- Affine transform (scale, translate, rotate, shear)
- ColorJitter
- GaussNoise or GaussianBlur (p=0.2)
- GridDistortion (p=0.2) for edge accuracy
- ImageNet normalization

**Model**

UNet++ with EfficientNet-B2 encoder and SCSE (Squeeze-and-Channel Squeeze-and-Excitation) decoder attention via `segmentation_models_pytorch`.

**Loss Function**

```python
loss = 0.6 × TverskyLoss(alpha=0.3, beta=0.7) + 0.4 × FocalLoss(gamma=2.0)
```

Tversky with `beta=0.7` penalizes false negatives more heavily, improving boundary recall on thin pet edges.

**Optimizer & Scheduler**

- AdamW (`lr=3e-4`, `weight_decay=1e-4`)
- 3-epoch linear warmup followed by cosine annealing to `eta_min=1e-6`
- Gradient clipping (`max_norm=1.0`) for training stability

---

## Training Configuration

### Classification

| Parameter | Phase 1 | Phase 2 |
|---|---|---|
| Backbone | Frozen | Unfrozen |
| Learning Rate | 2e-4 | 5e-5 |
| Optimizer | AdamW | AdamW |
| Scheduler | CosineAnnealingLR | CosineAnnealingLR |
| Batch Size | 32 | 32 |
| Max Epochs | 10 | 10 |
| Early Stopping Patience | 3 | 2 |
| Loss | CrossEntropy (ls=0.1) | CrossEntropy (ls=0.1) |

### Segmentation

| Parameter | Value |
|---|---|
| Input Size | 256 × 256 |
| Batch Size | 16 |
| Learning Rate | 3e-4 |
| Optimizer | AdamW |
| Scheduler | LinearWarmup (3 ep) + CosineAnnealing |
| Max Epochs | 15 |
| Early Stopping Patience | 4 |
| Loss | 0.6 × Tversky + 0.4 × Focal |
| Gradient Clip | 1.0 |

---

## Evaluation

### Classification Metrics

- Accuracy (top-1)
- Weighted Precision, Recall, F1-score
- Per-class accuracy (sorted worst → best)
- Full classification report
- Confusion matrix (18×18 visualization)

### Segmentation Metrics

- IoU (Intersection over Union)
- Dice Score (F1)
- Pixel Accuracy
- Precision and Recall
- All metrics computed with `smp.metrics` using micro-reduction

---

## Gradio Inference App

The project includes a Gradio interface combining both models for live inference:

**Inputs:** A single pet image (cat or dog)

**Outputs:**
- Top-5 breed predictions with confidence scores (`gr.Label`)
- Segmentation overlay (green highlight on pet, dimmed background)
- Confidence heatmap (RdYlGn colormap of sigmoid probabilities)

```python
demo.launch(debug=True)
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/sushantkothari/Cat-and-Dog-Breed-Classification-and-Segmentation-Using-EfficientNet-and-UNetPlusPlus.git
cd Cat-and-Dog-Breed-Classification-and-Segmentation-Using-EfficientNet-and-UNetPlusPlus
```

### Install Dependencies

```bash
pip install torch torchvision segmentation-models-pytorch albumentations \
            scikit-learn matplotlib gradio tqdm
```

---

## Usage

1. Open `Pet_Breed_Classification_and_Segmentation_System.ipynb` in Google Colab.
2. Mount Google Drive and place `oxford-iiit-pet.zip` in your Drive.
3. Run cells in order:
   - Environment check and dataset extraction
   - Classification: dataset, model, training, evaluation
   - Segmentation: dataset, model, training, evaluation
   - Gradio app launch
4. Upload any pet image in the Gradio interface to get breed predictions and segmentation output.

**Pretrained Weights**

- `best_model.pth` — Best classification checkpoint (included in repo)
- `best_seg_model.pth` — Best segmentation checkpoint (generated during training)

---

## Repository Structure

```
├── Pet_Breed_Classification_and_Segmentation_System.ipynb  # Main notebook
├── best_model.pth                                           # Classification weights
├── test_image.jpeg                                          # Sample test image
├── Results.png                                              # Results visualization
├── README.md
└── LICENSE
```

---

## Technology Stack

- Python
- PyTorch
- torchvision
- segmentation-models-pytorch
- Albumentations
- scikit-learn
- Matplotlib
- Gradio
- Google Colab

---

## Engineering Principles

- Mixed precision training (AMP) for GPU efficiency
- Gradient clipping for stable segmentation training
- Label smoothing for better-calibrated classification
- Tversky + Focal loss combination for boundary-sensitive segmentation
- SCSE decoder attention for refined spatial feature recalibration
- LR warmup to prevent early divergence in segmentation training
- Best-model checkpointing with early stopping for both tasks
- Modular notebook structure for easy experimentation

---

## Potential Extensions

- Multi-label classification (breed + species simultaneously)
- Instance segmentation with Mask R-CNN
- EfficientNet-B4/B7 for higher classification accuracy
- TensorRT or ONNX export for deployment
- Flask/FastAPI REST API wrapping the Gradio inference function
- Grad-CAM visualizations for classification explainability
- Mobile deployment via TorchScript or TFLite conversion

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Author

**Sushant Kothari**  
[GitHub](https://github.com/sushantkothari)
