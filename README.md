# Ocular-LLM-Explanability

This repository demonstrates how to generate **saliency maps** for ophthalmic (retinal) images using a **Vision Transformer** (ViT) model enhanced with **Layer-wise Relevance Propagation (LRP)**. These saliency maps, when combined with the original retinal images, can be fed into a Large Language Model (LLM) to provide visually guided explanations for diagnoses. Ultimately, our goal is to leverage these maps to guide LLMs in generating detailed explanations for retinal foundation model diagnoses.

## Contents

1. **`Transformer-Explainability/`**  
   A fork/clone of the [Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability) project by Hila Chefer et al., released under the MIT License. This project provides the LRP modules for ViTs (including `ViT_explanation_generator.py`).

2. **`RETFound_MAE/`**  
   A fork/modified version of the [RETFound](https://github.com/rmapho/RETFound) repository, which utilizes masked autoencoders for retinal image analysis. We have modified the ViT code (specifically, `models_vit_update.py`) to incorporate LRP logic into all transformer blocks.

3. **`ocular-llm-explanability/`** (this repository)  
   - **`generate_saliency_regression.py`** for single-class (regression) saliency map generation. This script produces a saliency map and prediction for each image, as well as one overall saliency map showing consistently important regions.
   - **`generate_saliency_multiclass.py`** for multi-class classification. This script generates per-image saliency maps and predictions, along with an overall saliency map for each predicted class. (If no images are predicted for a particular class, a message is printed and that class is skipped.)

## Main Modifications

- **LRP Integration:**  
  We integrated LRP modules from **Transformer-Explainability** into the ViT architecture from **RETFound** by modifying each transformer block to support backward propagation of relevance.

- **Saliency Map Generation:**  
  For each image, the following steps are performed:
  1. **Inference:**  
     The model predicts either a regression value or a class label. Predictions (and associated probabilities) are saved in a CSV file along with the image name.
  2. **LRP Attributions:**  
     LRP is applied to generate attributions for the final prediction. These attributions are resized from 14×14 to 224×224.
  3. **Normalization & Thresholding:**  
     The saliency maps are normalized to the range [0, 1] and can optionally be thresholded using Otsu's method.
  4. **Visualization:**  
     The normalized saliency map is blended with the original image to create a heatmap visualization.

- **Overall Saliency Map:**  
  We aggregate a binary mask (derived by selecting the top 10% of salient pixels) across multiple images. For regression, a single overall saliency map is produced; for multi-class classification, separate overall maps are generated for each predicted class. If no images are predicted for a class, a message is printed and that class is skipped.

## How to Use

### 1. Clone the Repositories

Ensure you have the following components available at the specified paths:

- **Transformer-Explainability:**  
  Clone this repository from [Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability).

- **RETFound_MAE:**  
  Clone or fork the [RETFound](https://github.com/rmapho/RETFound) repository, then copy the modified `models_vit_update.py` (provided in this repo) into your RETFound_MAE folder.

Make sure the **Transformer-Explainability** folder is placed in parallel with the **RETFound_MAE** folder. Update the `BASELINE_PATH` in `generate_saliency_regression.py`, `generate_saliency_multiclass.py`, and `models_vit_update.py` to point to your local Transformer-Explainability folder.

### 2. Install Dependencies

You can either:

- Create a conda environment for RETFound, install its dependencies, then add the Transformer-Explainability dependencies to the same environment.
- Or, simply run:
  ```bash
  pip install -r requirements.txt
  ```
  to install all required dependencies for ocular-llm-explanability.

### 3. Run the Saliency Scripts

#### Single-Class Regression

To generate saliency maps for a regression task, run:
```bash
python generate_saliency_regression.py \
    --checkpoint_path /path/to/checkpoint-best.pth \
    --input_folder /path/to/test_images \
    --gpu_ids 0 \
    --use_thresholding
```
This script will:
- Load the ViT model with LRP.
- Process each image in the specified folder.
- Save processed images, individual saliency maps, an overall saliency map, and a CSV file with predictions.

#### Multi-Class Classification

First, update the `class_labels` dictionary in `generate_saliency_multiclass.py` to match the classes predicted by your model. For example:
```python
# Example for 5 classes:
class_labels = {
    0: 'anormal',
    1: 'bmilddr',
    2: 'cmoderatedr',
    3: 'dseveredr',
    4: 'eprolifedr'
}
```
Then run:
```bash
python generate_saliency_multiclass.py \
    --checkpoint_path /path/to/checkpoint-best.pth \
    --input_folder /path/to/test_images \
    --gpu_ids 0 \
    --use_thresholding \
    --num_classes 5 \
    --top_k 1
```
This script will generate:
- Saliency maps and predictions for each image.
- Overall saliency maps for each predicted class (skipping any class with no predictions).

---

## References and Licenses

- **Transformer-Explainability (MIT License):**  
  [https://github.com/hila-chefer/Transformer-Explainability](https://github.com/hila-chefer/Transformer-Explainability)  
  (We adapted the LRP logic from this repository.)

- **RETFound:**  
  [https://github.com/rmapho/RETFound](https://github.com/rmapho/RETFound)  
  (The base ViT architecture and image transforms are derived from RETFound.)

- **Ocular-LLM-Explanability:**  
  This repository is intended for research and educational purposes only. It is not a medical device and should not be used for clinical decision-making.

For any questions or contributions, please open an issue or contact the repository maintainers.
