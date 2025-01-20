import math
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
from functools import partial
from torchvision import transforms

# Add necessary paths
sys.path.append("/home/user/yatpan/ocular-llm-explainability/Transformer-Explainability")
from baselines.ViT.ViT_explanation_generator import LRP
from models_vit_update import vit_large_patch16_with_lrp as vit_large_patch16
from util.datasets import build_transform

# Set device based on parsed GPU IDs
parser = argparse.ArgumentParser()
parser.add_argument('--gpu_ids', type=str, default='4', help='Comma-separated GPU IDs to use, e.g., "0,1,2"')
parser.add_argument("--checkpoint_path", type=str, help="Path to the model checkpoint.")
parser.add_argument("--image_folder", type=str, help="Folder containing test images.")
parser.add_argument("--output_folder", type=str, help="Folder to save saliency maps and CSV results.")
parser.add_argument("--input_size", type=int, default=224, help="Input image size for model.")
parser.add_argument("--drop_rate", type=float, default=0.0, help="Dropout rate for the model.")
parser.add_argument("--global_pool", action="store_true", default=True, help="Use global pooling for the model.")
parser.add_argument("--num_classes", type=int, help="Number of classes in the dataset.")
parser.add_argument("--use_thresholding", action="store_true", help="Apply thresholding on saliency maps.")
args = parser.parse_args()

# Set to use GPU 6 and 7
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Experiment setup
experiment = {
    "test_image_folder": "/home/user/yatpan/RETFound_MAE/BRSET/a-brazilian-multilabel-ophthalmological-dataset-brset-1.0.1/test_10_percent/images",
    "checkpoint_path": "/home/user/yatpan/RETFound_MAE/Glaucoma_fundus/checkpoint-best.pth",
    "class_labels": {0: 'anormal_control', 1: 'bearly_glaucoma', 2: 'cadvanced_glaucoma'},
    "num_classes": 3,
    "output_folder": "/home/user/yatpan/RETFound_MAE/output/brset_gf/"
}

# Define output paths
saliency_output_folder = os.path.join(experiment["output_folder"], "saliency_maps")
predictions_csv_path = os.path.join(experiment["output_folder"], "predictions.csv")

# Seed setting
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    print(f"Random seed set to: {seed}")

# Load model
def load_model(checkpoint_path, input_size, num_classes, drop_rate, global_pool):
    # Initialize model without loading weights
    model = vit_large_patch16(
        img_size=input_size,
        num_classes=num_classes,
        drop_rate=drop_rate,
        global_pool=global_pool
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    try:
        model.load_state_dict(checkpoint['model'], strict=True)
    except RuntimeError:
        print("Warning: Strict loading failed, attempting to load model without strict")
        model.load_state_dict(checkpoint['model'], strict=False)

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    print('number of params (M): %.2f' % (n_parameters / 1.e6))

    model.eval()
    model.to(device)
    print(f"Model loaded from checkpoint: {checkpoint_path}")
    
    return model

# Build transform
transform = build_transform(is_train=False, args=args)

# Inference on a single image
def infer(model, image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0).to(device)

    with torch.set_grad_enabled(True):
        output = model(image)
        prediction_softmax = torch.nn.functional.softmax(output, dim=1)
        _, predicted_index = torch.max(prediction_softmax, 1)

    # Return predicted index and the entire probability vector
    return predicted_index.item(), prediction_softmax.detach().cpu().numpy().flatten()

# Function to generate heatmap from LRP attributions
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

# Generate Saliency Map with Blending
def generate_visualization(original_image, attribution_generator, class_index=None, use_thresholding=False):
    transformer_attribution = attribution_generator.generate_LRP(
        original_image.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index)

    if transformer_attribution is None:
        raise ValueError("LRP attribution generation failed.")

    transformer_attribution = transformer_attribution.detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = F.interpolate(transformer_attribution, scale_factor=16, mode='bicubic').reshape(224, 224).cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    
    if use_thresholding:
        transformer_attribution = (transformer_attribution * 255).astype(np.uint8)
        _, transformer_attribution = cv2.threshold(transformer_attribution, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        transformer_attribution[transformer_attribution == 255] = 1

    image_transformer_attribution = original_image.permute(1, 2, 0).cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    
    return vis

# Process folder for inference and saliency maps
def process_folder(model, attribution_generator, folder_path, class_labels, saliency_output_folder, predictions_csv_path, use_thresholding=False):
    # Ensure the output directory exists
    if not os.path.exists(saliency_output_folder):
        os.makedirs(saliency_output_folder)

    results = []

    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file_name)

                # Inference
                pred_label, pred_prob = infer(model, img_path)
                pred_class_label = class_labels.get(pred_label, 'unknown')
                
                # Append the results for CSV
                results.append([file_name, pred_class_label, pred_prob.tolist()])

                # Generate Saliency Map
                input_image = transform(Image.open(img_path)).to(device)
                saliency_map = generate_visualization(input_image, attribution_generator, class_index=pred_label, use_thresholding=use_thresholding)
                
                # Save the saliency map with the same name as the image
                saliency_file_path = os.path.join(saliency_output_folder, f"{os.path.splitext(file_name)[0]}_saliency.png")
                plt.imsave(saliency_file_path, saliency_map, cmap='jet')

    # Save results to predictions CSV
    df = pd.DataFrame(results, columns=["Image ID", "Predicted Class", "Probabilities"])
    df.to_csv(predictions_csv_path, index=False)
    print(f"Results saved to {predictions_csv_path}")

# Main function
def main():
    set_seed(42)
    model = load_model(experiment["checkpoint_path"], args.input_size, experiment["num_classes"], args.drop_rate, args.global_pool)
    attribution_generator = LRP(model)
    process_folder(model, attribution_generator, experiment["test_image_folder"], experiment["class_labels"], saliency_output_folder, predictions_csv_path, use_thresholding=args.use_thresholding)

if __name__ == "__main__":
    main()
