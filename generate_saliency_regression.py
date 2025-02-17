"""
Copyright (c) Yating Pan, University of Zurich, Department of Computational Linguistics

Portions of this code are adapted from:
  - Transformer-Explainability (https://github.com/hila-chefer/Transformer-Explainability/), 
    under the MIT License, Copyright (c) 2021 Hila Chefer
  - RETFound (https://github.com/rmapho/RETFound), 
    under its respective license (Apache-2.0 or as specified).

This script demonstrates how to generate saliency maps for a single-class (regression) Vision Transformer (ViT) model that has been modified to include Layer-wise Relevance Propagation (LRP). The ViT model here is based on RETFound_MAE (for ophthalmic image analysis), with LRP logic integrated from Transformer-Explainability.

Usage:
  python generate_saliency_regression.py --checkpoint_path /path/to/checkpoint.pth \
      --input_folder /path/to/images --gpu_ids 0 --use_thresholding

Command-line arguments include:
  --checkpoint_path: Path to the trained model checkpoint
  --input_folder:    Folder containing images to analyze
  --input_size:      Image size (default 224)
  --drop_rate:       Dropout rate (default 0.0)
  --global_pool:     Whether to use global pooling (default True)
  --use_thresholding Whether to apply thresholding (Otsu) on saliency maps

Please refer to the accompanying README for more details.
"""


import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

# Set the baseline to the path where the Transformer-Explainability is cloned
# Ideally, Transformer-Explainability should be cloned at the same level as RETFound_MAE
BASELINE_PATH = "/data/yapan/Transformer-Explainability"
if BASELINE_PATH not in sys.path:
    sys.path.insert(0, BASELINE_PATH)
    
# Import the model from RETFound_MAE, here models_vit_update it an updated version of RETFound ViT model to add LRP(Layer-wise Relevance Propagation) in all classes of ViT
from models_vit_update import vit_large_patch16_with_lrp as vit_large_patch16
from models_vit_update import compute_rollout_attention

# Import the image transformation function from RETFound_MAE, here we don't use the ImageNet processing in Transformer-Explainability, we use the same transformation as in RETFound_MAE
from util.datasets import build_transform

# Import LRP from Transformer-Explainability, here we use LRP to generate saliency maps
from baselines.ViT.ViT_explanation_generator import LRP

# The original Transformer-Explainability code generates saliency maps for each predicted class (multi-class classification). Here we generate the saliency map for single class regression.
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', type=str, default='0', help='Comma-separated GPU IDs, e.g., "0,1,2"')
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint.")
parser.add_argument("--input_folder", type=str, required=True, help="Path to input image folder.")
parser.add_argument("--input_size", type=int, default=224, help="Input image size for model.")
parser.add_argument("--drop_rate", type=float, default=0.0, help="Dropout rate.")
parser.add_argument("--global_pool", action="store_true", default=True, help="Use global pooling for the model.")
parser.add_argument("--use_thresholding", action="store_true", help="Apply thresholding on saliency maps.")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Output folders, where the processed images, saliency maps, and predictions will be saved
output_folder = os.path.join(args.input_folder, "outputs")
saliency_output_folder = os.path.join(output_folder, "saliency_maps")
processed_image_folder = os.path.join(output_folder, "processed_images")
os.makedirs(saliency_output_folder, exist_ok=True)
os.makedirs(processed_image_folder, exist_ok=True)
print(f"Output folder created: {output_folder}")

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    print(f"Random seed set to: {seed}")

def load_model(checkpoint_path):
    model = vit_large_patch16(
        img_size=args.input_size,
        num_classes=1,  # set to 1 for regression task
        drop_rate=args.drop_rate,
        global_pool=args.global_pool
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval().to(device)
    print(f"Model loaded from {checkpoint_path}")
    return model

# Load the image transformation function from RETFound_MAE
transform = build_transform(is_train=False, args=args)

# Inference function to predict the value from an image
def infer(model, image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
    predicted_age = output.item()
    return predicted_age

def show_cam_on_image(img, mask):
    """
    This function is copied from the Transformer-Explainability code at https://colab.research.google.com/github/hila-chefer/Transformer-Explainability/blob/main/Transformer_explainability.ipynb#scrollTo=ZPbx6CIHEl08
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def generate_visualization(original_image, attribution_generator, use_thresholding=False, return_raw=True):
    """
    This function is modified from the Transformer-Explainability code at https://colab.research.google.com/github/hila-chefer/Transformer-Explainability/blob/main/Transformer_explainability.ipynb#scrollTo=ZPbx6CIHEl08
    """
    transformer_attribution = attribution_generator.generate_LRP(
        original_image.unsqueeze(0).to(device),
        method="transformer_attribution"
    )
    if transformer_attribution is None:
        raise ValueError("LRP attribution generation failed.")

    # Resize from 14x14 -> 224x224
    transformer_attribution = transformer_attribution.detach().reshape(1, 1, 14, 14)
    transformer_attribution = F.interpolate(
        transformer_attribution, scale_factor=16, mode='bilinear', align_corners=True
    ).reshape(224, 224).cpu().numpy()

    # Normalize [0,1], here we add eps to avoid division by zero
    eps = 1e-8
    transformer_attribution = (
        transformer_attribution - transformer_attribution.min()
    ) / (transformer_attribution.max() - transformer_attribution.min() + eps)

    # (Optional) Thresholding with Otsu
    if use_thresholding:
        attn_255 = (transformer_attribution * 255).astype(np.uint8)
        _, thresh_map = cv2.threshold(
            attn_255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        transformer_attribution = (thresh_map == 255).astype(np.float32) # here we threshold the saliency map instead of using continuous values

    # Convert image back to HWC in [0,1]
    img_np = original_image.permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + eps)

    # Blend
    vis = show_cam_on_image(img_np, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

    if return_raw:
        return vis, transformer_attribution
    else:
        return vis

def process_input_folder(model, attribution_generator, input_folder):
    results = []
    image_accumulator = np.zeros((224, 224, 3), dtype=np.float32)
    mask_accumulator = np.zeros((224, 224), dtype=np.float32)
    count = 0

    # 1) Let's pick a reference image at the start
    #    (here, we just pick the first .jpg or .png we find).
    reference_image_np = None
    for tmp in os.listdir(input_folder):
        if tmp.lower().endswith(('.png', '.jpg', '.jpeg')):
            ref_path = os.path.join(input_folder, tmp)
            pil_ref = Image.open(ref_path).convert("RGB")
            processed_ref = transform(pil_ref)  # 3×224×224
            # Convert to [H×W×3]
            reference_image_np = processed_ref.permute(1,2,0).cpu().numpy()
            # Normalize [0..1]
            eps = 1e-8
            reference_image_np = (reference_image_np - reference_image_np.min()) / \
                                 (reference_image_np.max() - reference_image_np.min() + eps)
            break

    for img_name in tqdm(os.listdir(input_folder), desc="Processing images"):
        img_path = os.path.join(input_folder, img_name)
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        predicted_age = infer(model, img_path)

        # Transform + save a processed copy
        pil_image = Image.open(img_path).convert("RGB")
        processed_image = transform(pil_image).to(device)
        processed_image_np = processed_image.permute(1, 2, 0).cpu().numpy()
        eps = 1e-8
        processed_image_np = (processed_image_np - processed_image_np.min()) / \
                             (processed_image_np.max() - processed_image_np.min() + eps)
        processed_image_file = os.path.join(
            processed_image_folder,
            f"{os.path.splitext(img_name)[0]}_processed.png"
        )
        plt.imsave(processed_image_file, processed_image_np)

        # Generate saliency + raw map
        saliency_map, raw_attribution = generate_visualization(
            processed_image,
            attribution_generator,
            use_thresholding=args.use_thresholding,
            return_raw=True
        )
        saliency_map_file = os.path.join(
            saliency_output_folder,
            f"{os.path.splitext(img_name)[0]}_saliency.png"
        )
        plt.imsave(saliency_map_file, saliency_map, cmap='jet')

        results.append([img_name, predicted_age])

        # For the overall map, we keep track of top-K% mask
        top_percent = 0.1
        h, w = raw_attribution.shape
        flat = raw_attribution.flatten()
        top_k = int(top_percent * h * w)
        idx = np.argpartition(flat, -top_k)[-top_k:]
        mask = np.zeros_like(flat, dtype=np.float32)
        mask[idx] = 1.0
        mask = mask.reshape(h, w)

        mask_accumulator += mask
        count += 1

    if count > 0:
        # fraction of images that highlight each pixel
        fraction_mask = mask_accumulator / count

        # 2) Instead of using avg_image, overlay on reference_image_np
        #    If reference_image_np is None, you can fallback to an average image.
        if reference_image_np is None:
            # fallback: average image
            reference_image_np = image_accumulator / count

        # Make final overlay
        overall_vis = show_cam_on_image(reference_image_np, fraction_mask)
        overall_vis = np.uint8(255 * overall_vis)
        overall_vis = cv2.cvtColor(np.array(overall_vis), cv2.COLOR_RGB2BGR)

        overall_saliency_path = os.path.join(saliency_output_folder, "overall_saliency.png")
        plt.imsave(overall_saliency_path, overall_vis, cmap='jet')
        print(f"Overall saliency map saved to {overall_saliency_path}")
    else:
        print("No images processed for overall saliency map.")

    # Save predictions
    results_df = pd.DataFrame(results, columns=["Image_Name", "Predicted_Age"])
    predictions_csv = os.path.join(output_folder, "predictions.csv")
    results_df.to_csv(predictions_csv, index=False)
    print(f"Predictions saved to {predictions_csv}")


def main():
    set_seed(42)
    model = load_model(args.checkpoint_path)
    attribution_generator = LRP(model)
    process_input_folder(model, attribution_generator, args.input_folder)

if __name__ == "__main__":
    main()
