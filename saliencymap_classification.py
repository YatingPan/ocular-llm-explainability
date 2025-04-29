#!/usr/bin/env python
"""
Multi-Class Saliency Map Generator for RETFound
===============================================

Copyright (c) Yating Pan, University of Zurich, Department of Computational Linguistics

Portions of this code are adapted from:
  - Transformer-Explainability (https://github.com/hila-chefer/Transformer-Explainability/), 
    under the MIT License, Copyright (c) 2021 Hila Chefer
  - RETFound (https://github.com/rmapho/RETFound), 
    under its respective license (Apache-2.0 or as specified).

This script generates saliency maps for a multi-class classification Vision Transformer 
(ViT) model with Layer-wise Relevance Propagation (LRP) with enhanced visualization.

For an input folder of retina fundus images, this script generates:
- Individual saliency maps for each image and class
- Overall saliency maps (pixel-level and patch-level) averaged across all images per class
- Predictions CSV file with predicted classes and probabilities
- Explanation figures combining all visualization types

Usage:
python saliencymap-multiclass.py \
    --checkpoint_path /path/to/checkpoint-best.pth \
    --input_folder /path/to/test_images \
    --num_classes 5 \
    --top_k 1
"""

import os
import sys
import argparse
import traceback

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.interpolate import griddata
from matplotlib.colors import LinearSegmentedColormap
from collections import defaultdict

# Set path to Transformer-Explainability
BASELINE_PATH = "/data/JH/yapan/ocular-llm-explainability/Transformer-Explainability"
if BASELINE_PATH not in sys.path:
    sys.path.insert(0, BASELINE_PATH)


# RETFound imports
from RETFound_MAE.models_vit_lrp import RETFound_mae
from RETFound_MAE.util.datasets import build_transform
from baselines.ViT.ViT_explanation_generator import LRP

# ----------------------
# Setup and Configuration
# ----------------------

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate saliency maps for RETFound multi-class models")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--input_folder", type=str, required=True, help="Input folder of images")
    parser.add_argument("--reference_image", type=str, default=None, 
                        help="Path to a reference eye image for overlays")
    parser.add_argument("--input_size", type=int, default=224, help="Input size")
    parser.add_argument("--drop_rate", type=float, default=0.0, help="Dropout rate for the model")
    parser.add_argument("--global_pool", action="store_true", default=True, 
                        help="Use global pooling for the model")
    parser.add_argument("--num_classes", type=int, default=5, 
                        help="Number of classes that the model predicts")
    parser.add_argument("--class_name", type=str, default="class", 
                        help="Class name for the output folder (e.g., 'dr', 'glaucoma')")
    parser.add_argument("--use_thresholding", action="store_true", 
                        help="Apply thresholding (Otsu) on saliency maps")
    parser.add_argument("--top_k", type=int, default=1, 
                        help="Number of top predictions to generate saliency maps for each image")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    parser.add_argument("--alpha", type=float, default=0.7, 
                        help="Transparency factor for high importance areas (0.0-1.0)")
    parser.add_argument("--method", type=str, default="transformer_attribution", 
                        choices=["transformer_attribution", "rollout"], 
                        help="Method to generate saliency maps")
    return parser.parse_args()


def setup_device():
    """Setup and return the appropriate device (always use first GPU if available)"""
    if torch.cuda.is_available():
        print(f"CUDA is available with {torch.cuda.device_count()} devices")
        print(f"Using device: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda:0")  # Always use the first GPU
    else:
        print("CUDA is NOT available, using CPU")
        return torch.device("cpu")


def setup_folders(args):
    """Create output folders"""
    # Setup output folders
    output_folder = os.path.join(os.path.dirname(args.input_folder), f"{args.class_name}_saliency")
    
    # Create folder structure
    folders = {
        "main": output_folder,
        "saliency": os.path.join(output_folder, "saliency_maps"),
        "processed": os.path.join(output_folder, "processed_inputs"),
        "overall": os.path.join(output_folder, "overall")
    }
    
    # Create directories
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)
    
    return folders


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    print(f"Random seed set to: {seed}")


# Custom class labels - modify these for your specific dataset
def get_class_labels(num_classes):
    """Get dictionary of class labels"""
    # Example for DR grading (0-4):
    if num_classes == 5:
        return {
            0: 'No DR',
            1: 'Mild DR',
            2: 'Moderate DR',
            3: 'Severe DR',
            4: 'Proliferative DR'
        }
    # Default fallback
    return {i: f'class_{i}' for i in range(num_classes)}


# ----------------------
# Model and Data Loading
# ----------------------

def load_model(args, device):
    """Load RETFound model with LRP capabilities"""
    
    # Create model
    model = RETFound_mae(
        img_size=args.input_size,
        num_classes=args.num_classes,
        drop_rate=args.drop_rate,
        global_pool=args.global_pool
    )
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of params: {n_parameters / 1.e6:.2f}M")

    # Set to eval mode for inference
    model.eval()
    model.to(device)
    print(f"Model loaded from checkpoint: {args.checkpoint_path}")

    return model


def get_reference_image(args, transform):
    """Get a reference eye image for saliency overlays"""
    reference_path = args.reference_image
    input_folder = args.input_folder
    
    candidate_images = []
    
    # If a reference path is provided, try to use it first
    if reference_path and os.path.exists(reference_path):
        try:
            pil_image = Image.open(reference_path).convert("RGB")
            return transform(pil_image)
        except Exception as e:
            print(f"Error loading reference image: {e}")
    
    # Otherwise select a good quality image from input folder
    if input_folder:
        image_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        # Try to assess image quality to pick a good reference image
        for img_file in image_files[:min(10, len(image_files))]:
            try:
                img_path = os.path.join(input_folder, img_file)
                pil_image = Image.open(img_path).convert("RGB")
                processed = transform(pil_image)
                
                # Use std deviation as simple quality metric
                img_np = processed.permute(1, 2, 0).numpy()
                std_dev = np.std(img_np)
                candidate_images.append((processed, std_dev, img_file))
                
            except Exception as e:
                print(f"Error analyzing image {img_file}: {e}")
                continue
        
        # Select the image with highest standard deviation (more details)
        if candidate_images:
            # Sort by std_dev in descending order
            candidate_images.sort(key=lambda x: x[1], reverse=True)
            print(f"Selected {candidate_images[0][2]} as reference image (quality score: {candidate_images[0][1]:.4f})")
            return candidate_images[0][0]
    
    # Fallback to blank image
    print("Warning: Using blank image as reference. Results may not be optimal.")
    return torch.zeros(3, 224, 224)


# ----------------------
# Inference and Saliency Functions
# ----------------------

def inference(model, image):
    """
    Run inference for classification model
    
    Args:
        model: RETFound model
        image: Input image tensor
        
    Returns:
        tuple: (predicted class index, probability array)
    """
    x = model(image.unsqueeze(0))
    
    # Apply softmax
    prediction_softmax = F.softmax(x, dim=1)
    _, predicted_index = torch.max(prediction_softmax, 1)
    
    # Return predicted class and probabilities
    return predicted_index.item(), prediction_softmax.squeeze(0).detach().cpu().numpy()


def get_raw_attribution(image, attribution_generator, method, class_index=None,use_thresholding=False):
    """Generate attribution map for input image for a specific class"""
    # Generate LRP attribution
    transformer_attribution = attribution_generator.generate_LRP(
        image.unsqueeze(0).cuda(),
        method=method,
        index=class_index
    ).detach()
    
    if transformer_attribution is None:
        raise ValueError("LRP attribution generation failed.")

    # Reshape from patch space (14x14) to pixel space (224x224)
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    
    # Upscale to image dimensions
    transformer_attribution = F.interpolate(
        transformer_attribution, scale_factor=16, mode='bilinear'
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
        transformer_attribution = thresh_map / 255.0  # Convert back to [0,1]

    return transformer_attribution


def generate_patch_level_saliency(attribution_map, shape=(14, 14)):
    """Generate patch-level saliency from pixel-level attribution map"""
    h, w = shape
    patch_importance = np.zeros((h, w))
    
    # Downsample to patch level using average pooling
    for i in range(h):
        for j in range(w):
            patch_importance[i, j] = np.mean(attribution_map[i*16:(i+1)*16, j*16:(j+1)*16])
    
    # Normalize to [0,1] range
    eps = 1e-8
    patch_importance = (patch_importance - patch_importance.min()) / (
        patch_importance.max() - patch_importance.min() + eps
    )
            
    return patch_importance


# ----------------------
# Visualization Functions
# ----------------------

def overlay_saliency(img, mask, alpha=0.7, min_alpha=0.3, cmap_name="jet"):
    """Create heatmap visualization with balanced transparency"""
    # Get colormap and apply to mask
    cmap = plt.get_cmap(cmap_name)
    colored_mask = cmap(mask)
    
    # Create variable alpha channel (more important = more opaque)
    custom_alpha = min_alpha + mask * (alpha - min_alpha)
    
    # Create heatmap with custom alpha
    heatmap_rgba = np.zeros((mask.shape[0], mask.shape[1], 4))
    heatmap_rgba[..., :3] = colored_mask[..., :3]
    heatmap_rgba[..., 3] = custom_alpha
    
    # Convert to uint8 for OpenCV
    heatmap_uint8 = (heatmap_rgba * 255).astype(np.uint8)
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Blend images
    result = img_uint8.copy()
    alpha_channel = heatmap_uint8[..., 3:4] / 255.0
    for c in range(3):
        result[..., c] = (1 - alpha_channel[..., 0]) * img_uint8[..., c] + \
                       alpha_channel[..., 0] * heatmap_uint8[..., c]
    
    # Normalize to [0,1]
    result = result / 255.0
    return result


def generate_visualization(image, attribution):
    """Create visualization by overlaying attribution map on image"""
    # Convert image to [H,W,3] in [0..1]
    eps = 1e-8
    img_np = image.detach().permute(1, 2, 0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + eps)
    
    # Create heatmap with the colormap
    heatmap = cv2.applyColorMap(np.uint8(255 * attribution), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # Blend heatmap with original image
    cam = heatmap + np.float32(img_np)
    cam = cam / np.max(cam)
    vis = np.uint8(255 * cam)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    
    return vis


def create_explanation_figure(ref_img_path, pixel_saliency_path, discrete_patch_path, 
                              enhanced_patch_path, class_idx, class_label, output_folder, 
                              avg_prediction=None, std_prediction=None):
    """Create combined explanation figure with all saliency visualizations"""
    # Load all images
    ref_img = plt.imread(ref_img_path)
    pixel_saliency = plt.imread(pixel_saliency_path)
    discrete_patch = plt.imread(discrete_patch_path)
    enhanced_patch = plt.imread(enhanced_patch_path)
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Column titles
    titles = [
        "Original Fundus Image", 
        "Pixel-level Saliency", 
        "Discrete Patch Saliency", 
        "Enhanced Patch Saliency"
    ]
    
    # Add all images to the figure
    for i, (img, title) in enumerate(zip(
        [ref_img, pixel_saliency, discrete_patch, enhanced_patch], titles)):
        axes[i].imshow(img)
        axes[i].set_title(title, fontsize=12)
        axes[i].axis('off')
    
    # Add main title with class information
    title = f"Class {class_idx} ({class_label}) Saliency Maps"
    if avg_prediction is not None and std_prediction is not None:
        title += f" (Avg: {avg_prediction:.2f}, Std: {std_prediction:.2f})"
    plt.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    # Save the figure
    explanation_path = os.path.join(output_folder, f"class_{class_idx}_{class_label}_explanation_figure.png")
    plt.savefig(explanation_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Explanation figure for class {class_idx} saved to {explanation_path}")
    return explanation_path


def generate_class_saliency_maps(class_idx, class_label, accumulator, reference_image, output_folder, args):
    """Generate all saliency visualizations for a specific class"""
    if accumulator["count"] == 0:
        print(f"[Class {class_idx}] No images predicted as this class. Skipping visualizations.")
        return
    
    print(f"\nGenerating visualizations for class {class_idx} ({class_label})...")
    
    # Create class-specific folder
    class_folder = os.path.join(output_folder, f"class_{class_idx}_{class_label}")
    os.makedirs(class_folder, exist_ok=True)
    
    eps = 1e-8

    # Get reference image
    ref_img_np = reference_image.detach().cpu().permute(1, 2, 0).numpy()
    ref_img_np = (ref_img_np - ref_img_np.min()) / (ref_img_np.max() - ref_img_np.min() + eps)
    
    # Save the reference image
    ref_img_path = os.path.join(class_folder, "reference_image.png")
    plt.imsave(ref_img_path, ref_img_np)
    
    # Normalize the mask accumulator and apply smoothing
    fraction_mask = accumulator["mask_accumulator"] / accumulator["count"]
    smoothed_mask = cv2.GaussianBlur(fraction_mask, (5, 5), 0)
    smoothed_mask = (smoothed_mask - smoothed_mask.min()) / (smoothed_mask.max() - smoothed_mask.min() + eps)
    
    # Calculate average raw attribution map if available
    if len(accumulator["raw_attr_maps"]) > 0:
        raw_maps = np.stack(accumulator["raw_attr_maps"], axis=0)
        avg_attr = np.mean(raw_maps, axis=0)
        avg_attr = (avg_attr - avg_attr.min()) / (avg_attr.max() - avg_attr.min() + eps)
    else:
        # Use smoothed mask if no raw maps available
        avg_attr = smoothed_mask
    
    # Get average prediction and standard deviation
    if len(accumulator["predictions"]) > 0:
        avg_prediction = np.mean(accumulator["predictions"])
        std_prediction = np.std(accumulator["predictions"])
    else:
        avg_prediction = None
        std_prediction = None
    
    # Common figure settings
    figsize = (8, 8)
    dpi = 100
    
    # 1. Pixel-level saliency with reference image
    ref_img_vis = overlay_saliency(ref_img_np, smoothed_mask, alpha=0.7, min_alpha=0.3)
    pixel_saliency_path = os.path.join(class_folder, f"pixel_saliency.png")
    plt.imsave(pixel_saliency_path, ref_img_vis)
    
    
    # 2. Generate patch-level saliency maps
    patch_importance = generate_patch_level_saliency(avg_attr)
    
    # 2a. Discrete patch visualization
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(ref_img_np)
    
    # Create patch visualization with discrete blocks
    h, w = ref_img_np.shape[:2]
    patch_size_h, patch_size_w = h // 14, w // 14
    
    for i in range(14):
        for j in range(14):
            # Get color from colormap
            color = plt.cm.get_cmap("jet")(patch_importance[i, j])
            
            # Apply balanced transparency
            min_alpha = 0.3
            alpha_value = min_alpha + patch_importance[i, j] * (args.alpha - min_alpha)
            
            # Draw rectangle
            rect = plt.Rectangle(
                (j*patch_size_w, i*patch_size_h),
                patch_size_w, patch_size_h,
                color=color[:3],
                alpha=alpha_value,
                linewidth=0.5 if patch_importance[i, j] > 0.5 else 0
            )
            plt.gca().add_patch(rect)
    
    plt.axis('off')
    discrete_patch_path = os.path.join(class_folder, "discrete_patch_saliency.png")
    plt.savefig(discrete_patch_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # 2b. Enhanced/Smooth patch visualization
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(ref_img_np)
    
    # Create smooth visualization using interpolation
    xi, yi = np.meshgrid(np.linspace(0, 1, 14), np.linspace(0, 1, 14))
    xi_upsampled = np.linspace(0, 1, w)
    yi_upsampled = np.linspace(0, 1, h)
    Xi, Yi = np.meshgrid(xi_upsampled, yi_upsampled)
    
    # Interpolate patch importance values
    patch_points = np.column_stack((yi.flatten(), xi.flatten()))
    patch_values = patch_importance.flatten()
    upsampled_importance = griddata(patch_points, patch_values, (Yi, Xi), method='cubic')
    
    # Plot with variable alpha
    alpha_values = min_alpha + upsampled_importance * (args.alpha - min_alpha)
    plt.imshow(upsampled_importance, cmap="jet", alpha=alpha_values,
              extent=(0, w, h, 0), interpolation='bilinear')
    
    # Add subtle grid lines
    for i in range(1, 14):
        plt.axhline(y=i*patch_size_h, color='white', linestyle='-', alpha=0.2, linewidth=0.5)
        plt.axvline(x=i*patch_size_w, color='white', linestyle='-', alpha=0.2, linewidth=0.5)
    
    plt.axis('off')
    enhanced_patch_path = os.path.join(class_folder, "enhanced_patch_saliency.png")
    plt.savefig(enhanced_patch_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Create combined explanation figure
    create_explanation_figure(
        ref_img_path,
        pixel_saliency_path,
        discrete_patch_path,
        enhanced_patch_path,
        class_idx,
        class_label,
        class_folder,
        avg_prediction,
        std_prediction
    )
    
    print(f"All visualizations for class {class_idx} saved to: {class_folder}")


# ----------------------
# Main Process Functions
# ----------------------

def process_images(model, attribution_generator, args, transform, folders, class_labels):
    """Process all images in the input folder"""
    results = []
    
    # Dictionary to store reference images for each class
    reference_images = {}
    
    # Create a separate accumulator for each class
    class_accumulators = {
        c: {
            "image_accumulator": np.zeros((224, 224, 3), dtype=np.float32),
            "mask_accumulator": np.zeros((224, 224), dtype=np.float32),
            "raw_attr_maps": [],
            "count": 0,
            "predictions": []
        }
        for c in range(args.num_classes)
    }
    
    # Get list of image files
    image_files = [f for f in os.listdir(args.input_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    if not image_files:
        print(f"No image files found in {args.input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Process each image
    for img_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(args.input_folder, img_name)
        
        try:
            # Load and transform image
            pil_image = Image.open(img_path).convert("RGB")
            processed_image = transform(pil_image).to(args.device)
            
            # Save processed image
            processed_image_np = processed_image.detach().permute(1, 2, 0).cpu().numpy()
            eps = 1e-8
            processed_image_np = (processed_image_np - processed_image_np.min()) / (
                processed_image_np.max() - processed_image_np.min() + eps
            )
            processed_path = os.path.join(
                folders["processed"], 
                f"{os.path.splitext(img_name)[0]}_processed.png"
            )
            plt.imsave(processed_path, processed_image_np)
            
            # Run inference
            pred_label, pred_prob = inference(model, processed_image)
            pred_class_label = class_labels.get(pred_label, f'class_{pred_label}')
            
            if args.debug:
                print(f"Image: {img_name}, Predicted: {pred_label} ({pred_class_label})")
            
            # Generate saliency maps for top-K predicted classes
            top_k_indices = pred_prob.argsort()[::-1][:args.top_k] if args.num_classes > 1 else [0]
            
            for class_idx in top_k_indices:
                try:
                    # Get attribution map for this class
                    raw_attr = get_raw_attribution(
                        processed_image,
                        attribution_generator,
                        method=args.method,
                        class_index=class_idx,
                        use_thresholding=args.use_thresholding
                    )
                    
                    # Save individual saliency map
                    saliency_map = generate_visualization(processed_image, raw_attr)
                    
                    out_path = os.path.join(
                        folders["saliency"],
                        f"{os.path.splitext(img_name)[0]}_class_{class_idx}_saliency.png"
                    )
                    
                    plt.imsave(out_path, saliency_map)
                    
                except Exception as e:
                    print(f"Error generating saliency for {img_name}, class {class_idx}: {e}")
                    if args.debug:
                        traceback.print_exc()
            
            # For overall accumulators, use top-1 predicted class
            try:
                # Get attribution map for the predicted class
                raw_attr_pred = get_raw_attribution(
                    processed_image,
                    attribution_generator,
                    method=args.method,
                    class_index=pred_label,
                    use_thresholding=False
                )
                
                # Update accumulator for the predicted class
                class_acc = class_accumulators[pred_label]
                class_acc["image_accumulator"] += processed_image_np
                class_acc["count"] += 1
                class_acc["raw_attr_maps"].append(raw_attr_pred)
                class_acc["predictions"].append(pred_prob[pred_label])
                
                # Store a reference image for each class if not already set or higher quality
                img_quality = np.std(processed_image_np)
                if pred_label not in reference_images or img_quality > reference_images[pred_label][1]:
                    reference_images[pred_label] = (processed_image, img_quality)
                
                # Create binary mask of top 10% pixels
                top_percent = 0.1
                h, w = raw_attr_pred.shape
                flat = raw_attr_pred.flatten()
                top_k_pixels = int(top_percent * h * w)
                idx = np.argpartition(flat, -top_k_pixels)[-top_k_pixels:]
                mask = np.zeros_like(flat, dtype=np.float32)
                mask[idx] = 1.0
                mask = mask.reshape(h, w)
                
                # Add mask to accumulator
                class_acc["mask_accumulator"] += mask
                
            except Exception as e:
                print(f"Error processing accumulator for {img_name}: {e}")
                if args.debug:
                    traceback.print_exc()
            
            # Store prediction info
            results.append([
                img_name, 
                pred_label, 
                pred_prob.tolist() if isinstance(pred_prob, np.ndarray) else pred_prob
            ])
            
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")
            if args.debug:
                traceback.print_exc()
    
    # Save predictions to CSV
    if results:
        results_df = pd.DataFrame(results, columns=["Image_Name", "Predicted_Class", "Class_Probabilities"])
        predictions_file = os.path.join(folders["main"], "predictions.csv")
        results_df.to_csv(predictions_file, index=False)
        print(f"Saved predictions to {predictions_file}")
    
    # Generate saliency visualizations for each class
    for class_idx in range(args.num_classes):
        # Check if this class has any samples
        if class_accumulators[class_idx]["count"] > 0:
            # Use the corresponding reference image if available, otherwise use first image
            reference_img = reference_images.get(class_idx, (None, 0))[0]
            if reference_img is None and reference_images:
                # Fallback to any available reference image
                reference_img = next(iter(reference_images.values()))[0]
            
            # Generate saliency maps for this class
            generate_class_saliency_maps(
                class_idx, 
                class_labels.get(class_idx, f'class_{class_idx}'),
                class_accumulators[class_idx],
                reference_img,
                folders["overall"],
                args
            )
    
    return class_accumulators


# ----------------------
# Main Function
# ----------------------

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()
    
    # Setup device ONCE and store in args
    args.device = setup_device()
    print(f"Using device: {args.device}")
    
    # Create output folders
    folders = setup_folders(args)
    
    print(f"\n=== Enhanced Multi-Class Saliency Map Generator ===\n")
    
    # Set random seed
    set_seed(42)
    
    # Load transform function
    transform = build_transform(is_train=False, args=args)
    
    # Get class labels
    class_labels = get_class_labels(args.num_classes)
    
    # Load model
    model = load_model(args, args.device)
    
    # Create attribution generator
    attribution_generator = LRP(model)
    print(f"Using attribution generator: {type(attribution_generator).__name__}")
    
    # Process images
    process_images(model, attribution_generator, args, transform, folders, class_labels)
    
    print("\n=== Saliency map generation complete! ===")
    print(f"Results saved to: {folders['main']}")
    print(f"Overall results saved to: {folders['overall']}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()
        sys.exit(1)