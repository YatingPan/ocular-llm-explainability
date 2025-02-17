"""
Copyright (c) Yating Pan, University of Zurich, Department of Computational Linguistics

Portions of this code are adapted from:
  - Transformer-Explainability (https://github.com/hila-chefer/Transformer-Explainability/), 
    under the MIT License, Copyright (c) 2021 Hila Chefer
  - RETFound (https://github.com/rmapho/RETFound), 
    under its respective license (Apache-2.0 or as specified).

This script demonstrates how to generate saliency maps for a multi- class classification Vision Transformer (ViT) model that has been modified to include Layer-wise Relevance Propagation (LRP). The ViT model here is based on RETFound_MAE (for ophthalmic image analysis), with LRP logic integrated from Transformer-Explainability.

Usage:
python generate_saliency_multiclass.py \
    --checkpoint_path /path/to/checkpoint-best.pth \
    --input_folder /path/to/test_images \
    --gpu_ids 0 \
    --use_thresholding \
    --num_classes 5 \
    --top_k 1

Command-line arguments include:
  --checkpoint_path: Path to the trained model checkpoint
  --input_folder:    Folder containing images to analyze
  --input_size:      Image size (default 224)
  --drop_rate:       Dropout rate (default 0.0)
  --global_pool:     Whether to use global pooling (default True)
  --use_thresholding Whether to apply thresholding (Otsu) on saliency maps
  --num_classes:     Number of classes that the model predicts
  --top_k:           Number of top predictions to generate saliency maps for each image

Please refer to the accompanying README for more details.
"""


import os
import sys
import torch
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch.nn.functional as F
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict

# Set the baseline to the path where the Transformer-Explainability is cloned
# Ideally, Transformer-Explainability should be cloned at the same level as RETFound_MAE
sys.path.append("/data/yapan/Transformer-Explainability")

# Import the model from RETFound_MAE, here models_vit_update it an updated version of RETFound ViT model to add LRP(Layer-wise Relevance Propagation) in all classes of ViT
from models_vit_update import vit_large_patch16_with_lrp as vit_large_patch16

# Import the image transformation function from RETFound_MAE, here we don't use the ImageNet processing in Transformer-Explainability, we use the same transformation as in RETFound_MAE
from util.datasets import build_transform

# Import LRP from Transformer-Explainability, here we use LRP to generate saliency maps
from baselines.ViT.ViT_explanation_generator import LRP

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', type=str, default='0', help='Comma-separated GPU IDs to use, e.g., "0,1,2"')
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint.")
parser.add_argument("--test_folder", type=str, required=True, help="Folder containing images to analyze.")
parser.add_argument("--input_size", type=int, default=224, help="Input image size for model.")
parser.add_argument("--drop_rate", type=float, default=0.0, help="Dropout rate for the model.")
parser.add_argument("--global_pool", action="store_true", default=True, help="Use global pooling for the model.")
parser.add_argument("--num_classes", type=int, default=5, help="Number of classes that the model predicts.")
parser.add_argument("--use_thresholding", action="store_true", help="Apply thresholding (Otsu) on saliency maps.")
parser.add_argument("--top_k", type=int, default=1, help="Number of top predictions to generate saliency maps for each image.")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Output folders, where the processed images, saliency maps, and predictions will be saved
output_folder = os.path.join(args.input_folder, "outputs")
saliency_output_folder = os.path.join(args.output_folder, "saliency_maps")
processed_image_folder = os.path.join(args.output_folder, "processed_inputs")
os.makedirs(saliency_output_folder, exist_ok=True)
os.makedirs(processed_image_folder, exist_ok=True)

# Example: 5 classes
# Change the class labels as needed
class_labels = {
    0: 'anormal',
    1: 'bmilddr',
    2: 'cmoderatedr',
    3: 'dseveredr',
    4: 'eprolifedr'
}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    print(f"Random seed set to: {seed}")

def load_model(checkpoint_path, input_size, drop_rate, global_pool, num_classes):
    model = vit_large_patch16(
        img_size=input_size,
        num_classes=num_classes,
        drop_rate=drop_rate,
        global_pool=global_pool
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'], strict=False)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params (M): {n_parameters / 1.e6:.2f}")

    model.eval()
    model.to(device)
    print(f"Model loaded from checkpoint: {checkpoint_path}")

    return model

# Load the image transformation function from RETFound_MAE
transform = build_transform(is_train=False, args=args)

def infer(model, image):
    """
    Returns:
      pred_label: the top-1 predicted class index
      pred_prob: a 1D array of softmax probabilities for all classes
    """
    image = image.unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)  # shape: (1, num_classes)
        prediction_softmax = F.softmax(output, dim=1)  # shape: (1, num_classes)
        _, predicted_index = torch.max(prediction_softmax, 1)
    return predicted_index.item(), prediction_softmax.squeeze(0).cpu().numpy()

def show_cam_on_image(img, mask):
    """
    This function is copied from the Transformer-Explainability code at https://colab.research.google.com/github/hila-chefer/Transformer-Explainability/blob/main/Transformer_explainability.ipynb#scrollTo=ZPbx6CIHEl08
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def get_raw_attribution(original_image, attribution_generator, class_index=None, use_thresholding=False):
    """
    This function is modified from the Transformer-Explainability code at https://colab.research.google.com/github/hila-chefer/Transformer-Explainability/blob/main/Transformer_explainability.ipynb#scrollTo=ZPbx6CIHEl08
    """
    # Generate LRP
    transformer_attribution = attribution_generator.generate_LRP(
        original_image.unsqueeze(0).to(device),
        method="transformer_attribution",
        index=class_index
    )
    if transformer_attribution is None:
        raise ValueError("LRP attribution generation failed.")

    # Reshape from 14x14 -> 224x224
    transformer_attribution = transformer_attribution.detach().reshape(1, 1, 14, 14)
    transformer_attribution = F.interpolate(
        transformer_attribution, scale_factor=16, mode='bilinear', align_corners=True
    ).reshape(224, 224).cpu().numpy()

    # Normalize to [0..1]
    eps = 1e-8
    transformer_attribution = (
        transformer_attribution - transformer_attribution.min()
    ) / (transformer_attribution.max() - transformer_attribution.min() + eps)

    # Optional Otsu thresholding
    if use_thresholding:
        attn_255 = (transformer_attribution * 255).astype(np.uint8)
        _, thresh_map = cv2.threshold(attn_255, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        transformer_attribution = (thresh_map == 255).astype(np.float32)

    return transformer_attribution

def generate_visualization(original_image, raw_attribution):
    """
    This function is modified from the Transformer-Explainability code at https://colab.research.google.com/github/hila-chefer/Transformer-Explainability/blob/main/Transformer_explainability.ipynb#scrollTo=ZPbx6CIHEl08
    """
    eps = 1e-8
    # Convert image to [H,W,3] in [0..1]
    img_np = original_image.permute(1,2,0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + eps)

    vis = show_cam_on_image(img_np, raw_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    return vis

def process_test_folder(
    model,
    attribution_generator,
    class_labels,
    test_folder,
    saliency_output_folder,
    processed_image_folder,
    use_thresholding=False
):
    """
    For each image:
      1) Run inference -> get top-1 predicted class (plus softmax).
      2) Save per-image saliency for top-K predicted classes.
      3) For the aggregator, only store the top-1 predicted class
         in the accumulators. 
    At the end, create an overall saliency for each class that has at least one image.
    """
    results = []

    # We create a dictionary of accumulators, one for each class
    class_accumulators = {
        c: {
            "image_accumulator": np.zeros((224,224,3), dtype=np.float32),
            "mask_accumulator": np.zeros((224,224), dtype=np.float32),
            "count": 0
        }
        for c in range(args.num_classes)
    }

    for img_name in tqdm(os.listdir(test_folder), desc="Processing images"):
        img_path = os.path.join(test_folder, img_name)
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        # Load and transform image
        pil_image = Image.open(img_path).convert("RGB")
        processed_image = transform(pil_image)  # shape: 3×224×224

        # Save processed image for reference
        processed_image_np = processed_image.permute(1,2,0).cpu().numpy()
        eps = 1e-8
        processed_image_np = (processed_image_np - processed_image_np.min()) / (
            processed_image_np.max() - processed_image_np.min() + eps
        )
        processed_image_file = os.path.join(
            processed_image_folder, f"{img_name}_processed.png"
        )
        plt.imsave(processed_image_file, processed_image_np)

        # Inference
        pred_label, pred_prob = infer(model, processed_image)
        pred_class_label = class_labels.get(pred_label, 'unknown')

        # Save saliency maps for top-K predicted classes
        # e.g. if top_k=2, we show saliency for the top 2 classes
        top_k_indices = pred_prob.argsort()[::-1][:args.top_k]
        for class_idx in top_k_indices:
            raw_attr = get_raw_attribution(
                processed_image,
                attribution_generator,
                class_index=class_idx,
                use_thresholding=use_thresholding
            )
            saliency_map_bgr = generate_visualization(processed_image, raw_attr)
            out_path = os.path.join(
                saliency_output_folder,
                f"{img_name}_class_{class_idx}_saliency.png"
            )
            plt.imsave(out_path, saliency_map_bgr, cmap='jet')

        # For the "overall" saliency aggregator, 
        # we only store the top-1 predicted class.
        raw_attr_pred = get_raw_attribution(
            processed_image,
            attribution_generator,
            class_index=pred_label,
            use_thresholding=False  # typically no thresholding here, we do top-percent below
        )

        # Accumulate for that predicted class
        class_acc = class_accumulators[pred_label]
        class_acc["image_accumulator"] += processed_image_np
        class_acc["count"] += 1

        # Convert raw_attribution to a top-10% binary mask
        top_percent = 0.1
        h, w = raw_attr_pred.shape
        flat = raw_attr_pred.flatten()
        top_k_pixels = int(top_percent * h * w)
        idx = np.argpartition(flat, -top_k_pixels)[-top_k_pixels:]
        mask = np.zeros_like(flat, dtype=np.float32)
        mask[idx] = 1.0
        mask = mask.reshape(h, w)

        class_acc["mask_accumulator"] += mask

        # Save predictions info
        results.append([img_name, pred_label, pred_prob.tolist()])

    # Create a DataFrame of predictions
    results_df = pd.DataFrame(results, columns=["Image_Name", "Predicted_Indices", "Predicted_Probabilities"])
    results_df.to_csv(os.path.join(args.output_folder, "predictions.csv"), index=False)

    # Now, for each class that has at least one image, create an overall saliency
    for c in range(args.num_classes):
        count_c = class_accumulators[c]["count"]
        if count_c > 0:
            # fraction of images that highlight each pixel
            fraction_mask = class_accumulators[c]["mask_accumulator"] / count_c
            avg_image = class_accumulators[c]["image_accumulator"] / count_c

            # Overlay fraction_mask on the average image
            overall_vis = show_cam_on_image(avg_image, fraction_mask)
            overall_vis = np.uint8(255 * overall_vis)
            overall_vis = cv2.cvtColor(np.array(overall_vis), cv2.COLOR_RGB2BGR)

            # Save
            class_label_str = class_labels.get(c, f"class_{c}")
            overall_saliency_path = os.path.join(
                saliency_output_folder,
                f"overall_saliency_class_{c}_{class_label_str}.png"
            )
            plt.imsave(overall_saliency_path, overall_vis, cmap='jet')
            print(f"[Class {c}] Overall saliency map saved to {overall_saliency_path}")
        else:
            print(f"[Class {c}] No images predicted as this class. Skipping overall saliency.")

def main():
    set_seed(42)
    model = load_model(
        args.checkpoint_path,
        args.input_size,
        args.drop_rate,
        args.global_pool,
        args.num_classes
    )
    attribution_generator = LRP(model)

    process_test_folder(
        model,
        attribution_generator,
        class_labels,
        args.test_folder,
        saliency_output_folder,
        processed_image_folder,
        args.use_thresholding
    )

if __name__ == "__main__":
    main()
