#!/usr/bin/env python
"""
Copyright (c) Yating Pan, University of Zurich, 2025

RETFound Regression Saliency Map Generator

This script generates saliency maps for RETFound model trained on regression tasks using 
Layer-wise Relevance Propagation (LRP) method from Transformer-Explainability. 

For an input folder of retina fundus images, this script generates:

- Individual saliency maps for each image
- Overall saliency maps (pixel-level and patch-level) averaged across all images
- Predictions CSV file with predicted values
- Histogram of prediction distribution

Usage:
python saliency_regression.py \
    --checkpoint_path /path/to/checkpoint-best.pth \
    --input_folder /path/to/test_images \
    --method transformer_attribution
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

# Set path to Transformer-Explainability -> Update this to your local path
BASELINE_PATH = "/data/JH/yapan/ocular-llm-explainability/Transformer-Explainability"
if BASELINE_PATH not in sys.path:
    sys.path.insert(0, BASELINE_PATH)

# RETFound imports
from RETFound_MAE.models_vit_lrp import RETFound_mae
from RETFound_MAE.util.datasets import build_transform
from RETFound_MAE.regression_lrp import RegressionLRP

# ----------------------
# Setup and Configuration
# ----------------------

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate saliency maps for RETFound regression models")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--input_folder", type=str, required=True, help="Input folder of retina images")
    parser.add_argument("--reference_image", type=str, default=None, 
                        help="Path to a reference eye image for overall saliency map")
    parser.add_argument("--input_size", type=int, default=224, help="Input size")
    parser.add_argument("--drop_rate", type=float, default=0.0, help="Dropout rate for the model")
    parser.add_argument("--drop_path", type=float, default=0.2, 
                        help="Drop path rate (stochastic depth), default 0.2")
    parser.add_argument("--global_pool", action="store_true", default=True, 
                        help="Use global pooling for the model")
    parser.add_argument("--output_dim", type=int, default=1, 
                        help="Output dimension for the regression model (usually 1)")
    parser.add_argument("--use_thresholding", action="store_true", 
                        help="Apply thresholding (Otsu) on saliency maps")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    parser.add_argument("--output_folder", type=str, default=None, help="Custom output folder")
    parser.add_argument("--metric_name", type=str, default="Value", 
                        help="Name of the metric being predicted (e.g., 'Angle', 'Age')")
    parser.add_argument("--alpha", type=float, default=0.6, 
                        help="Transparency factor for heatmap overlay (0.0-1.0)")
    parser.add_argument("--method", type=str, default="transformer_attribution", 
                        choices=["transformer_attribution", "rollout"], 
                        help="Method to generate saliency maps")
    return parser.parse_args()


def setup_device():
    """Setup and return the appropriate device (always use first GPU if available)"""
    if torch.cuda.is_available():
        return torch.device("cuda:0")  # Always use the first GPU
    else:
        print("CUDA is NOT available, using CPU")
        return torch.device("cpu")


def setup_folders(args):
    """Create output folders"""
    if args.output_folder:
        output_folder = args.output_folder
    else:
        output_folder = os.path.join(os.path.dirname(args.input_folder), 
                                     f"{args.metric_name.lower()}_regression")
    
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


# ----------------------
# Model and Data Loading
# ----------------------

def load_model(args, device):
    """Load RETFound model with LRP capabilities"""
    
    # Create model
    model = RETFound_mae(
        img_size=args.input_size,
        num_classes=args.output_dim,
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path,
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

def inference(model, image):
    """Run inference on a single image"""
    output = model(image.unsqueeze(0))
    return output.squeeze().item()

def get_reference_image(args, transform):
    """Get a reference eye image for saliency overlays"""
    # Try specified reference image first
    reference_path = args.reference_image
    input_folder = args.input_folder
    
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
        
        candidates = []
        for img_file in image_files[:min(10, len(image_files))]:
            try:
                img_path = os.path.join(input_folder, img_file)
                pil_image = Image.open(img_path).convert("RGB")
                processed = transform(pil_image)
                
                # Use std deviation as simple quality metric
                img_np = processed.permute(1, 2, 0).numpy()
                std_dev = np.std(img_np)
                candidates.append((processed, std_dev, img_file))
            except Exception:
                continue
        
        # Select the image with highest standard deviation (more detail)
        if candidates:
            candidates.sort(key=lambda x: x[1], reverse=True)
            best_image = candidates[0]
            print(f"Selected {best_image[2]} as reference (quality: {best_image[1]:.4f})")
            return best_image[0]
    
    # Fallback to blank image
    print("Warning: Using blank image as reference. Results may not be optimal.")
    return torch.zeros(3, 224, 224)


# ----------------------
# Saliency Map Generation
# ----------------------

def get_raw_attribution(image, attribution_generator, method, use_thresholding):
    """Generate attribution map for regression task"""
    # Generate LRP attribution
    transformer_attribution = attribution_generator.generate_LRP(
        image.unsqueeze(0).cuda(),
        method=method,
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

def overlay_saliency(img, mask, alpha=0.6, min_alpha=0.3):
    """Create heatmap visualization with balanced transparency"""
    # Get colormap and apply to mask
    cmap = plt.get_cmap("jet")
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


def create_explanation_figure(ref_img_path, pixel_saliency_path, discrete_path, 
                              enhanced_path, metric_name, output_folder, 
                              avg_prediction, std_prediction):
    """Create combined explanation figure with all saliency visualizations"""
    # Load all the pre-saved images
    ref_img = plt.imread(ref_img_path)
    pixel_saliency = plt.imread(pixel_saliency_path)
    discrete_patch = plt.imread(discrete_path)
    enhanced_patch = plt.imread(enhanced_path)
    
    # Create a single row figure with 4 columns
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
    
    # Add main title
    plt.suptitle(
        f"{metric_name} Prediction Saliency Maps (Avg: {avg_prediction:.2f}, Std: {std_prediction:.2f})", 
        fontsize=14
    )
    
    plt.tight_layout()
    
    # Save the combined figure
    explanation_path = os.path.join(output_folder, f"{metric_name.lower()}_explanation_figure.png")
    plt.savefig(explanation_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Explanation figure saved to {explanation_path}")
    return explanation_path


def generate_saliency_visualizations(reference_image, accumulator, overall_folder, metric_name,avg_prediction, std_prediction):
    """Generate all saliency visualizations (pixel-level and patch-level)"""
    eps = 1e-8
    
    # Get reference image for visualizations
    ref_img_np = reference_image.detach().cpu().permute(1, 2, 0).numpy()
    ref_img_np = (ref_img_np - ref_img_np.min()) / (ref_img_np.max() - ref_img_np.min() + eps)
    
    # Save reference image
    ref_img_path = os.path.join(overall_folder, "reference_image.png")
    plt.imsave(ref_img_path, ref_img_np)
    
    # Average attribution maps
    all_maps = np.stack(accumulator["raw_attr_maps"], axis=0)
    avg_attr = np.mean(all_maps, axis=0)
    avg_attr = (avg_attr - avg_attr.min()) / (avg_attr.max() - avg_attr.min() + eps)
    
    # Calculate frequency mask (how often each pixel is in top 10%)
    fraction_mask = accumulator["mask_accumulator"] / accumulator["count"]
    
    # Apply Gaussian smoothing for better visualization
    smoothed_mask = cv2.GaussianBlur(fraction_mask, (5, 5), 0)
    smoothed_mask = (smoothed_mask - smoothed_mask.min()) / (smoothed_mask.max() - smoothed_mask.min() + eps)
    
    # Common figure settings
    figsize = (8, 8)
    dpi = 100
    
    # 1. Generate pixel-level saliency with reference image
    ref_img_vis = overlay_saliency(ref_img_np, smoothed_mask, alpha=0.7, min_alpha=0.3)
    pixel_saliency_path = os.path.join(overall_folder, f"pixel_saliency.png")
    plt.imsave(pixel_saliency_path, ref_img_vis)
    
    # 2. Generate patch-level saliency
    patch_importance = generate_patch_level_saliency(avg_attr)
    
    # 2a. Discrete patch visualization
    h, w = ref_img_np.shape[:2]
    patch_size_h, patch_size_w = h // 14, w // 14
    
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(ref_img_np)
    
    # Create discrete patches
    for i in range(14):
        for j in range(14):
            # Get color from colormap
            color = plt.cm.get_cmap("jet")(patch_importance[i, j])
            
            # Calculate alpha based on importance
            alpha_value = 0.3 + patch_importance[i, j] * 0.4
            
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
    discrete_patch_path = os.path.join(overall_folder, f"discrete_patch_saliency.png")
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
    alpha_values = 0.3 + upsampled_importance * 0.4
    plt.imshow(upsampled_importance, cmap="jet", alpha=alpha_values,
              extent=(0, w, h, 0), interpolation='bilinear')
    
    # Add subtle grid lines
    for i in range(1, 14):
        plt.axhline(y=i*patch_size_h, color='white', linestyle='-', alpha=0.2, linewidth=0.5)
        plt.axvline(x=i*patch_size_w, color='white', linestyle='-', alpha=0.2, linewidth=0.5)
    
    plt.axis('off')
    enhanced_patch_path = os.path.join(overall_folder, f"enhanced_patch_saliency.png")
    plt.savefig(enhanced_patch_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Create combined explanation figure
    create_explanation_figure(
        ref_img_path,
        pixel_saliency_path,
        discrete_patch_path,
        enhanced_patch_path,
        metric_name,
        overall_folder,
        avg_prediction,
        std_prediction
    )


# ----------------------
# Main Process Functions
# ----------------------

def process_images(model, attribution_generator, args, transform, folders):
    """Process all images in the input folder"""
    results = []
    
    # Setup accumulator for averaging results
    accumulator = {
        "image_accumulator": np.zeros((224, 224, 3), dtype=np.float32),
        "raw_attr_maps": [],
        "mask_accumulator": np.zeros((224, 224), dtype=np.float32),
        "count": 0,
        "sum_predictions": 0,
        "predictions": []
    }
    
    # Get list of image files
    image_files = [f for f in os.listdir(args.input_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    if not image_files:
        print(f"No image files found in {args.input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Get reference image for visualizations
    reference_image = get_reference_image(args, transform)
    
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
            pred_value = inference(model, processed_image)
            
            if args.debug:
                print(f"Image: {img_name}, Predicted {args.metric_name}: {pred_value:.4f}")
            
            # Generate saliency map
            try:
                # Get raw attribution map
                raw_attr = get_raw_attribution(
                    processed_image,
                    attribution_generator,
                    method=args.method,
                    use_thresholding=args.use_thresholding
                )
                
                # Save individual saliency map
                saliency_map = generate_visualization(processed_image, raw_attr)
                out_path = os.path.join(
                    folders["saliency"],
                    f"{os.path.splitext(img_name)[0]}_saliency.png"
                )
                plt.imsave(out_path, saliency_map)
                
                # Store for overall analysis
                accumulator["raw_attr_maps"].append(raw_attr)
                
                # Create top-10% binary mask
                top_percent = 0.1
                h, w = raw_attr.shape
                flat = raw_attr.flatten()
                top_k_pixels = int(top_percent * h * w)
                idx = np.argpartition(flat, -top_k_pixels)[-top_k_pixels:]
                mask = np.zeros_like(flat, dtype=np.float32)
                mask[idx] = 1.0
                mask = mask.reshape(h, w)
                
                # Add to accumulator
                accumulator["mask_accumulator"] += mask
                
            except Exception as e:
                print(f"Error generating saliency for {img_name}: {e}")
                if args.debug:
                    traceback.print_exc()
                continue
            
            # Update accumulator
            accumulator["image_accumulator"] += processed_image_np
            accumulator["count"] += 1
            accumulator["sum_predictions"] += pred_value
            accumulator["predictions"].append(pred_value)
            
            # Store result
            results.append([img_name, pred_value])
            
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")
            if args.debug:
                traceback.print_exc()
    
    # Save predictions to CSV
    if accumulator["count"] > 0:
        # Calculate statistics
        avg_prediction = accumulator["sum_predictions"] / accumulator["count"]
        predictions = np.array(accumulator["predictions"])
        std_prediction = np.std(predictions)
        
        # Save CSV results
        results_df = pd.DataFrame(results, columns=["Image_Name", f"Predicted_{args.metric_name}"])
        results_df.to_csv(os.path.join(folders["main"], "predictions.csv"), index=False)
        
        print(f"Average {args.metric_name}: {avg_prediction:.4f}, Std: {std_prediction:.4f}")
        
        # Generate overall saliency visualizations
        if accumulator["raw_attr_maps"]:
            generate_saliency_visualizations(
                reference_image, 
                accumulator, 
                folders["overall"],
                args.metric_name,
                avg_prediction,
                std_prediction
            )
        
        # Save histogram of predictions
        plt.figure(figsize=(10, 6))
        plt.hist(predictions, bins=20)
        plt.xlabel(f'Predicted {args.metric_name}')
        plt.ylabel('Count')
        plt.title(f'Distribution of {args.metric_name} Predictions (Mean: {avg_prediction:.2f}, Std: {std_prediction:.2f})')
        plt.savefig(os.path.join(folders["main"], "prediction_distribution.png"))
        plt.close()
    
    return accumulator


# ----------------------
# Main Function
# ----------------------

def main():
    """Main function"""
    # Parse arguments
    args = parse_args()

    # setup device
    args.device = setup_device()
    print(f"Using device: {args.device}")
    
    # Create output folders
    folders = setup_folders(args)
    
    print(f"\n=== RETFound Regression Saliency Map Generator - Using {args.method} method ===\n")
    print(f"Processing {args.metric_name} prediction model")
    
    # Set random seed
    set_seed(42)
    
    # Load transform function
    transform = build_transform(is_train=False, args=args)
    
    # Load model
    model = load_model(args, args.device)
    
    # Create attribution generator
    attribution_generator = RegressionLRP(model)
    print(f"Using attribution generator: {type(attribution_generator).__name__} with method: {args.method}")
    
    # Process images
    process_images(model, attribution_generator, args, transform, folders)
    
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