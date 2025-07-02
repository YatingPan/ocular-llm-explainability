"""
Copyright (c) Yating Pan, University of Zurich, Department of Computational Linguistics

Portions of this code are adapted from:
  - Transformer-Explainability (https://github.com/hila-chefer/Transformer-Explainability/), 
    under the MIT License, Copyright (c) 2021 Hila Chefer
  - RETFound (https://github.com/rmapho/RETFound), 
    under its respective license (Apache-2.0 or as specified).

This script demonstrates how to generate saliency maps for a multi-class classification 
Vision Transformer (ViT) model with Layer-wise Relevance Propagation (LRP) with enhanced
visualization techniques.

Usage:
python saliencymap_classification.py \
    --checkpoint_path /NVME/scratch/dave/VD_fold_1/fold1/checkpoint-best.pth \
    --input_folder /HDD/data/yating/200_img \
    --num_classes 1
"""

import os
import sys
import argparse
import traceback

# Set path to Transformer-Explainability
BASELINE_PATH = "/HDD/data/yating/ocular-llm-explainability/Transformer-Explainability"
if BASELINE_PATH not in sys.path:
    sys.path.insert(0, BASELINE_PATH)

parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', type=str, default='0', help='Comma-separated GPU IDs')
parser.add_argument("--checkpoint_path", type=str, default="/data/JH/yapan/RETFound_MAE/BRSET_checkpoints/dr/checkpoint-best.pth", help="Path to checkpoint")
parser.add_argument("--input_folder", type=str, default="/data/JH/yapan/SaliencyMap/explanation_survey", help="Input folder")
parser.add_argument("--reference_image", type=str, default=None, help="Path to a reference eye image for overlays")
parser.add_argument("--input_size", type=int, default=224, help="Input size")
parser.add_argument("--drop_rate", type=float, default=0.0, help="Dropout rate for the model.")
parser.add_argument("--global_pool", action="store_true", default=True, help="Use global pooling for the model.")
parser.add_argument("--num_classes", type=int, default=1, help="Number of classes that the model predicts.")
parser.add_argument("--use_thresholding", action="store_true", help="Apply thresholding (Otsu) on saliency maps.")
parser.add_argument("--top_k", type=int, default=1, help="Number of top predictions to generate saliency maps for each image.")
parser.add_argument("--debug", action="store_true", help="Print debug information.")
parser.add_argument("--alpha", type=float, default=0.7, help="Transparency factor for high importance areas (0.0-1.0)")
parser.add_argument("--min_alpha", type=float, default=0.3, help="Minimum transparency for low importance areas (0.0-1.0)")
parser.add_argument("--cmap", type=str, default="jet", help="Colormap for heatmaps (e.g., 'jet', 'viridis', 'inferno')")
parser.add_argument("--output_folder", type=str, default=None, help="Output folder for saliency maps")
args = parser.parse_args()

# Ensure CUDA_VISIBLE_DEVICES is set before importing torch
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

import torch
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from tqdm import tqdm
from collections import defaultdict
from scipy.interpolate import griddata

# Import the model from RETFound_MAE
from RETFound_MAE.models_vit_lrp import RETFound_mae
from baselines.ViT.ViT_explanation_generator import LRP

# Import the image transformation function from RETFound_MAE
from RETFound_MAE.util.datasets import build_transform

# Use CPU for debugging or GPU otherwise
device = torch.device("cpu" if args.debug else f"cuda:{args.gpu_ids.split(',')[0]}")
print(f"Using device: {device}")

# Output folders, where the processed images, saliency maps, and predictions will be saved
output_folder = args.output_folder if args.output_folder else os.path.join(args.input_folder, "d")
saliency_output_folder = os.path.join(output_folder, "saliency_maps")
processed_image_folder = os.path.join(output_folder, "processed_inputs")
overall_folder = os.path.join(output_folder, "overall")  # New folder for overall visualizations

os.makedirs(saliency_output_folder, exist_ok=True)
os.makedirs(processed_image_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)
os.makedirs(overall_folder, exist_ok=True)  # Create overall folder

# Example: Class labels (customize as needed)
class_labels = {
            0: 'No DR',
            1: 'Mild DR',
            2: 'Moderate DR',
            3: 'Severe DR',
            4: 'Proliferative DR'
}

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    print(f"Random seed set to: {seed}")

def load_model(checkpoint_path, input_size, drop_rate, global_pool, num_classes):
    model = RETFound_mae(
        img_size=input_size,
        num_classes=num_classes,
        drop_rate=drop_rate,
        global_pool=global_pool
    )
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device, weights_only=False)
    
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

# Load the image transformation function
transform = build_transform(is_train=False, args=args)

def infer(model, image):
    """
    Returns:
      pred_label: the top-1 predicted class index
      pred_prob: a 1D array of softmax probabilities for all classes
    """
    x = model(image.unsqueeze(0).cuda())
    
    # Apply softmax
    prediction_softmax = F.softmax(x, dim=1)
    _, predicted_index = torch.max(prediction_softmax, 1)
    
    # Detach before converting to numpy
    return predicted_index.item(), prediction_softmax.squeeze(0).detach().cpu().numpy()

def show_cam_on_image(img, mask):
    """
    Overlay heatmap on image
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def show_cam_on_image_with_balanced_transparency(img, mask, alpha=0.7, cmap_name="jet", min_alpha=0.3):
    """
    Create a heatmap visualization overlaid on the image with balanced transparency
    that keeps blue regions visible while showing underlying details
    
    Args:
        img: Normalized [H,W,C] image array
        mask: [H,W] attribution mask
        alpha: Blending factor for the highest importance areas (0.0-1.0)
        cmap_name: Colormap name from matplotlib
        min_alpha: Minimum alpha/opacity for the lowest importance areas
        
    Returns:
        Heatmap overlaid on the image with balanced transparency
    """
    # Get colormap
    cmap = plt.get_cmap(cmap_name)
    
    # Apply colormap to mask
    colored_mask = cmap(mask)
    
    # Create a custom alpha channel that varies gradually
    # - High importance areas (red) get the full specified alpha
    # - Low importance areas (blue) get a reduced but still visible alpha
    custom_alpha = min_alpha + mask * (alpha - min_alpha)
    
    # Create heatmap with custom alpha
    heatmap_rgba = np.zeros((mask.shape[0], mask.shape[1], 4))
    heatmap_rgba[..., :3] = colored_mask[..., :3]
    heatmap_rgba[..., 3] = custom_alpha
    
    # Convert to uint8 for OpenCV
    heatmap_uint8 = (heatmap_rgba * 255).astype(np.uint8)
    img_uint8 = (img * 255).astype(np.uint8)
    
    # Use OpenCV to blend images
    result = img_uint8.copy()
    alpha_channel = heatmap_uint8[..., 3:4] / 255.0
    for c in range(3):
        result[..., c] = (1 - alpha_channel[..., 0]) * img_uint8[..., c] + \
                         alpha_channel[..., 0] * heatmap_uint8[..., c]
    
    # Normalize to [0,1]
    result = result / 255.0
    
    return result

def get_raw_attribution(original_image, attribution_generator, class_index=None, use_thresholding=False):
    """
    Generate attribution map for input image
    """
    # Generate LRP
    transformer_attribution = attribution_generator.generate_LRP(
        original_image.unsqueeze(0).cuda(),
        method="transformer_attribution",
        index=class_index
    ).detach()
    
    if transformer_attribution is None:
        raise ValueError("LRP attribution generation failed.")

    # Reshape from 14x14 -> 224x224
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
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

def generate_visualization(original_image, raw_attribution):
    """
    Create visualization by overlaying attribution map on image
    """
    eps = 1e-8
    # Convert image to [H,W,3] in [0..1]
    img_np = original_image.detach().permute(1,2,0).cpu().numpy()
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + eps)
    vis = show_cam_on_image(img_np, raw_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    return vis

def generate_patch_level_saliency(attribution_map, shape=(14, 14)):
    """
    Generate patch-level saliency from a pixel-level attribution map
    
    Args:
        attribution_map: Pixel-level attribution map (224x224)
        shape: Output shape (default: 14x14 for ViT patches)
        
    Returns:
        Patch-level importance map
    """
    h, w = shape
    patch_importance = np.zeros((h, w))
    
    # Downsample to patch level using average pooling
    for i in range(h):
        for j in range(w):
            patch_importance[i, j] = np.mean(attribution_map[i*16:(i+1)*16, j*16:(j+1)*16])
    
    # Normalize the patch importance to [0,1] range
    eps = 1e-8
    patch_importance = (patch_importance - patch_importance.min()) / (
        patch_importance.max() - patch_importance.min() + eps
    )
            
    return patch_importance

def get_reference_image(reference_path=None, input_folder=None):
    """
    Get a reference eye image for overlays with improved quality
    
    Args:
        reference_path: Path to a specified reference image
        input_folder: Folder to get a reference sample from if no reference is specified
        
    Returns:
        Processed reference image, selected for high quality
    """
    candidate_images = []
    
    # If a reference path is provided, try to use it first
    if reference_path and os.path.exists(reference_path):
        try:
            pil_image = Image.open(reference_path).convert("RGB")
            return transform(pil_image)
        except Exception as e:
            print(f"Error loading reference image: {e}")
    
    # If no reference or error, select the best quality image from input folder
    if input_folder:
        image_files = [f for f in os.listdir(input_folder) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
        
        # Try to assess image quality to pick a good reference image
        for idx, img_file in enumerate(image_files[:min(10, len(image_files))]):  # Check up to 10 images
            try:
                img_path = os.path.join(input_folder, img_file)
                pil_image = Image.open(img_path).convert("RGB")
                processed = transform(pil_image)
                
                # Convert to numpy for analysis
                img_np = processed.permute(1, 2, 0).numpy()
                
                # Simple metric: standard deviation as a very basic "detail" measure
                # Higher std dev often correlates with more detail/contrast
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
            return candidate_images[0][0]  # Return the highest quality image
    
    # If all else fails, return a blank image
    print("Warning: Using blank image as reference. Results may not be optimal.")
    return torch.zeros(3, 224, 224)

def create_explanation_figure(
    ref_img_path,
    pixel_saliency_path,
    discrete_patch_path,
    enhanced_patch_path,
    class_idx,
    class_label,
    output_folder,
    avg_prediction=None,
    std_prediction=None
):
    """
    Create a row of 4 images showing the original image and all 3 saliency visualizations
    
    Args:
        ref_img_path: Path to the reference/original image
        pixel_saliency_path: Path to the pixel-level saliency image
        discrete_patch_path: Path to the discrete patch saliency image
        enhanced_patch_path: Path to the enhanced/smoothed patch saliency image
        class_idx: Class index
        class_label: Class label name
        output_folder: Output folder to save the figure
        avg_prediction: Average prediction value (optional)
        std_prediction: Standard deviation of predictions (optional)
    """
    # Load all the pre-saved images
    ref_img = plt.imread(ref_img_path)
    pixel_saliency = plt.imread(pixel_saliency_path)
    discrete_patch = plt.imread(discrete_patch_path)
    enhanced_patch = plt.imread(enhanced_patch_path)
    
    # Create a single row figure with 4 columns
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Column titles
    titles = [
        "Original Fundus Image", 
        "Pixel-level Saliency", 
        "Discrete Patch Saliency", 
        "Enhanced Patch Saliency"
    ]
    
    # 1. Original image
    axes[0].imshow(ref_img)
    axes[0].set_title(titles[0], fontsize=12)
    axes[0].axis('off')
    
    # 2. Pixel-level saliency
    axes[1].imshow(pixel_saliency)
    axes[1].set_title(titles[1], fontsize=12)
    axes[1].axis('off')
    
    # 3. Discrete patch saliency
    axes[2].imshow(discrete_patch)
    axes[2].set_title(titles[2], fontsize=12)
    axes[2].axis('off')
    
    # 4. Enhanced patch saliency
    axes[3].imshow(enhanced_patch)
    axes[3].set_title(titles[3], fontsize=12)
    axes[3].axis('off')
    
    # Add a main title with class information
    title = f"Class {class_idx} ({class_label}) Saliency Maps"
    if avg_prediction is not None and std_prediction is not None:
        title += f" (Avg: {avg_prediction:.2f}, Std: {std_prediction:.2f})"
    plt.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    # Save the explanation figure
    explanation_path = os.path.join(output_folder, f"class_{class_idx}_{class_label}_explanation_figure.png")
    plt.savefig(explanation_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Explanation figure for class {class_idx} saved to {explanation_path}")
    return explanation_path

def process_input_folder(
    model,
    attribution_generator,
    class_labels,
    input_folder,
    saliency_output_folder,
    processed_image_folder,
    overall_folder,
    use_thresholding=False,
    alpha=0.7,
    min_alpha=0.3,
    cmap_name="jet"
):
    """
    Process all images in the input folder
    """
    results = []

    # Create a dictionary of accumulators for each class
    class_accumulators = {
        c: {
            "image_accumulator": np.zeros((224,224,3), dtype=np.float32),
            "mask_accumulator": np.zeros((224,224), dtype=np.float32),
            "raw_attr_maps": [],  # Added for storing raw attribute maps
            "count": 0,
            "predictions": []  # Added for storing predictions
        }
        for c in range(args.num_classes)
    }

    # Get list of image files
    image_files = [f for f in os.listdir(input_folder) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))]
    
    if not image_files:
        print(f"No image files found in {input_folder}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    # Get reference images for each class (will be selected/updated during processing)
    reference_images = {}
    
    for img_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(input_folder, img_name)
        
        try:
            # Load and transform image
            pil_image = Image.open(img_path).convert("RGB")
            processed_image = transform(pil_image).to(device)

            # Save processed image for reference
            processed_image_np = processed_image.detach().permute(1,2,0).cpu().numpy()
            eps = 1e-8
            processed_image_np = (processed_image_np - processed_image_np.min()) / (
                processed_image_np.max() - processed_image_np.min() + eps
            )
            processed_image_file = os.path.join(
                processed_image_folder, f"{os.path.splitext(img_name)[0]}_processed.png"
            )
            plt.imsave(processed_image_file, processed_image_np)

            # Inference
            pred_label, pred_prob = infer(model, processed_image)
            pred_class_label = class_labels.get(pred_label, f'class_{pred_label}')
            
            if args.debug:
                print(f"Image: {img_name}, Predicted: {pred_label} ({pred_class_label})")

            # Save saliency maps for top-K predicted classes
            top_k_indices = pred_prob.argsort()[::-1][:args.top_k] if args.num_classes > 1 else [0]
            
            for class_idx in top_k_indices:
                try:
                    raw_attr = get_raw_attribution(
                        processed_image,
                        attribution_generator,
                        class_index=class_idx,
                        use_thresholding=use_thresholding
                    )
                    
                    saliency_map_bgr = generate_visualization(processed_image, raw_attr)
                    
                    out_path = os.path.join(
                        saliency_output_folder,
                        f"{os.path.splitext(img_name)[0]}_class_{class_idx}_saliency.png"
                    )
                    
                    plt.imsave(out_path, saliency_map_bgr)
                    
                    if args.debug:
                        print(f"Saved saliency map for class {class_idx} to {out_path}")
                        
                except Exception as e:
                    print(f"Error generating saliency for {img_name}, class {class_idx}: {e}")
                    if args.debug:
                        import traceback
                        traceback.print_exc()

            # For overall accumulators, use top-1 predicted class
            try:
                raw_attr_pred = get_raw_attribution(
                    processed_image,
                    attribution_generator,
                    class_index=pred_label,
                    use_thresholding=False
                )

                # Update accumulator
                class_acc = class_accumulators[pred_label]
                class_acc["image_accumulator"] += processed_image_np
                class_acc["count"] += 1
                class_acc["raw_attr_maps"].append(raw_attr_pred)  # Store raw attribution map
                class_acc["predictions"].append(pred_prob[pred_label])  # Store prediction probability

                # Store a reference image for each class if not already set or if this one is high quality
                if pred_label not in reference_images or np.std(processed_image_np) > reference_images[pred_label][1]:
                    reference_images[pred_label] = (processed_image, np.std(processed_image_np))

                # Convert to top-10% mask
                top_percent = 0.1
                h, w = raw_attr_pred.shape
                flat = raw_attr_pred.flatten()
                top_k_pixels = int(top_percent * h * w)
                idx = np.argpartition(flat, -top_k_pixels)[-top_k_pixels:]
                mask = np.zeros_like(flat, dtype=np.float32)
                mask[idx] = 1.0
                mask = mask.reshape(h, w)

                class_acc["mask_accumulator"] += mask
                
            except Exception as e:
                print(f"Error processing accumulator for {img_name}: {e}")
                if args.debug:
                    import traceback
                    traceback.print_exc()

            # Save prediction info
            results.append([img_name, pred_label, pred_prob.tolist() if isinstance(pred_prob, np.ndarray) else pred_prob])
            
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()

    # Save predictions
    results_df = pd.DataFrame(results, columns=["Image_Name", "Predicted_Indices", "Predicted_Probabilities"])
    predictions_file = os.path.join(output_folder, "predictions.csv")
    results_df.to_csv(predictions_file, index=False)
    print(f"Saved predictions to {predictions_file}")

    # Generate enhanced overall saliency maps for each class
    figsize = (8, 8)
    dpi = 100
    
    for c in range(args.num_classes):
        # Create class-specific folder
        class_overall_folder = os.path.join(overall_folder, f"class_{c}_{class_labels.get(c, 'unknown')}")
        os.makedirs(class_overall_folder, exist_ok=True)
        
        count_c = class_accumulators[c]["count"]
        if count_c > 0:
            print(f"\nGenerating enhanced visualizations for class {c} ({class_labels.get(c, 'unknown')})...")
            
            # Get the average image for this class
            avg_image = class_accumulators[c]["image_accumulator"] / count_c
            avg_image = (avg_image - avg_image.min()) / (avg_image.max() - avg_image.min() + 1e-8)
            
            # Save the average image
            avg_img_path = os.path.join(class_overall_folder, "average_image.png")
            plt.imsave(avg_img_path, avg_image)
            
            # Get reference image for this class
            if c in reference_images:
                ref_image = reference_images[c][0]
                ref_img_np = ref_image.detach().cpu().permute(1, 2, 0).numpy()
                ref_img_np = (ref_img_np - ref_img_np.min()) / (ref_img_np.max() - ref_img_np.min() + 1e-8)
            else:
                # Use average image if no reference image is available
                ref_img_np = avg_image
            
            # Save the reference image
            ref_img_path = os.path.join(class_overall_folder, "reference_image.png")
            plt.imsave(ref_img_path, ref_img_np)
            
            # Normalize the mask accumulator and apply smoothing
            fraction_mask = class_accumulators[c]["mask_accumulator"] / count_c
            smoothed_mask = cv2.GaussianBlur(fraction_mask, (5, 5), 0)
            smoothed_mask = (smoothed_mask - smoothed_mask.min()) / (smoothed_mask.max() - smoothed_mask.min() + 1e-8)
            
            # Calculate average raw attribution map if available
            if len(class_accumulators[c]["raw_attr_maps"]) > 0:
                raw_maps = np.stack(class_accumulators[c]["raw_attr_maps"], axis=0)
                avg_attr = np.mean(raw_maps, axis=0)
                avg_attr = (avg_attr - avg_attr.min()) / (avg_attr.max() - avg_attr.min() + 1e-8)
            else:
                # Use smoothed mask if no raw maps available
                avg_attr = smoothed_mask
            
            # Get average prediction and standard deviation
            if len(class_accumulators[c]["predictions"]) > 0:
                avg_prediction = np.mean(class_accumulators[c]["predictions"])
                std_prediction = np.std(class_accumulators[c]["predictions"])
            else:
                avg_prediction = None
                std_prediction = None
            
            # 1. PIXEL-LEVEL SALIENCY WITH BALANCED TRANSPARENCY USING REFERENCE IMAGE
            plt.figure(figsize=figsize, dpi=dpi)
            ref_img_vis = show_cam_on_image_with_balanced_transparency(
                ref_img_np, 
                smoothed_mask, 
                alpha=alpha, 
                cmap_name=cmap_name,
                min_alpha=min_alpha
            )
            plt.imshow(ref_img_vis)
            plt.axis('off')
            pixel_saliency_path = os.path.join(class_overall_folder, f"pixel_saliency_reference.png")
            plt.savefig(pixel_saliency_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # 2. PIXEL-LEVEL SALIENCY WITH BALANCED TRANSPARENCY USING AVERAGE IMAGE
            plt.figure(figsize=figsize, dpi=dpi)
            avg_img_vis = show_cam_on_image_with_balanced_transparency(
                avg_image, 
                smoothed_mask, 
                alpha=alpha, 
                cmap_name=cmap_name,
                min_alpha=min_alpha
            )
            plt.imshow(avg_img_vis)
            plt.axis('off')
            avg_pixel_saliency_path = os.path.join(class_overall_folder, f"pixel_saliency_average.png")
            plt.savefig(avg_pixel_saliency_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # 3. DISCRETE PATCH-LEVEL VISUALIZATION WITH BALANCED TRANSPARENCY
            patch_importance = generate_patch_level_saliency(avg_attr)
            plt.figure(figsize=figsize, dpi=dpi)
            plt.imshow(ref_img_np)  # Use reference image as background
            
            # Create patch visualization with discrete blocks and balanced transparency
            h, w = ref_img_np.shape[:2]
            patch_size_h, patch_size_w = h // 14, w // 14
            
            # Create a custom RGBA patch overlay
            for i in range(14):
                for j in range(14):
                    # Get color from colormap
                    color = plt.cm.get_cmap(cmap_name)(patch_importance[i, j])
                    
# Apply balanced transparency
                    alpha_value = min_alpha + patch_importance[i, j] * (alpha - min_alpha)  # Range from min_alpha to alpha
                    
                    # Draw a semi-transparent patch rectangle
                    rect = plt.Rectangle(
                        (j*patch_size_w, i*patch_size_h),
                        patch_size_w, patch_size_h,
                        color=color[:3],
                        alpha=alpha_value,
                        linewidth=0.5 if patch_importance[i, j] > 0.5 else 0
                    )
                    plt.gca().add_patch(rect)
            
            plt.axis('off')
            discrete_patch_path = os.path.join(class_overall_folder, f"discrete_patch_saliency.png")
            plt.savefig(discrete_patch_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # 4. ENHANCED PATCH VISUALIZATION WITH BALANCED TRANSPARENCY
            plt.figure(figsize=figsize, dpi=dpi)
            
            # Plot the reference eye image
            plt.imshow(ref_img_np)
            
            # Create a smoother patch visualization using interpolation
            # Start with a low-resolution version
            xi, yi = np.meshgrid(np.linspace(0, 1, 14), np.linspace(0, 1, 14))
            
            # Upsample to image dimensions
            xi_upsampled = np.linspace(0, 1, w)
            yi_upsampled = np.linspace(0, 1, h)
            Xi, Yi = np.meshgrid(xi_upsampled, yi_upsampled)
            
            # Interpolate the patch importance values
            patch_points = np.column_stack((yi.flatten(), xi.flatten()))
            patch_values = patch_importance.flatten()
            upsampled_importance = griddata(patch_points, patch_values, (Yi, Xi), method='cubic')
            
            # Plot with balanced alpha values
            alpha_values = min_alpha + upsampled_importance * (alpha - min_alpha)
            plt.imshow(upsampled_importance, cmap=cmap_name, alpha=alpha_values,
                      extent=(0, w, h, 0), interpolation='bilinear')
            
            # Add subtle grid lines to show patches
            for i in range(1, 14):
                plt.axhline(y=i*patch_size_h, color='white', linestyle='-', alpha=0.2, linewidth=0.5)
                plt.axvline(x=i*patch_size_w, color='white', linestyle='-', alpha=0.2, linewidth=0.5)
            
            plt.axis('off')
            enhanced_patch_path = os.path.join(class_overall_folder, f"enhanced_patch_saliency.png")
            plt.savefig(enhanced_patch_path, dpi=dpi, bbox_inches='tight', pad_inches=0)
            plt.close()
            
            # Create and save the explanation figure combining all visualizations
            create_explanation_figure(
                ref_img_path,
                pixel_saliency_path,
                discrete_patch_path,
                enhanced_patch_path,
                c,
                class_labels.get(c, f"class_{c}"),
                class_overall_folder,
                avg_prediction,
                std_prediction
            )
            
            print(f"All visualizations for class {c} saved to: {class_overall_folder}")
            
        else:
            print(f"[Class {c}] No images predicted as this class. Skipping visualizations.")

def main():
    print(f"\n=== Enhanced Multi-Class Saliency Map Generator ===\n")
    
    set_seed(42)
    
    # Load model
    model = load_model(
        args.checkpoint_path,
        args.input_size,
        args.drop_rate,
        args.global_pool,
        args.num_classes
    )
    
    # Use LRP for attribution generation
    attribution_generator = LRP(model)
    
    print(f"Using attribution generator: {type(attribution_generator).__name__}")
    
    # Process images
    process_input_folder(
        model,
        attribution_generator,
        class_labels,
        args.input_folder,
        saliency_output_folder,
        processed_image_folder,
        overall_folder,
        args.use_thresholding,
        args.alpha,
        args.min_alpha,
        args.cmap
    )
    
    print("\n=== Saliency map generation complete! ===")
    print(f"Results saved to: {output_folder}")
    print(f"Overall results saved to: {overall_folder}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Critical error: {e}")
        traceback.print_exc()
        sys.exit(1)