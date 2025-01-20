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
from sklearn.metrics import average_precision_score, label_ranking_average_precision_score, multilabel_confusion_matrix, roc_auc_score
import torch.nn.functional
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# The following experiments are defined to reproduce the RETFound diagnosis and saliency maps using its provided checkpoints and datasets
# To use this script, please update the test_image_folder paths and checkpoint paths to the paths on your local machine
experiments = [
    {
        "test_image_folder": "/home/user/yatpan/RETFound_MAE/IDRID/IDRiD_data/test",
        "checkpoint_path": "/home/user/yatpan/RETFound_MAE/IDRID/checkpoint-best.pth",
        "class_labels": {0: 'anoDR', 1: 'bmildDR', 2: 'cmoderateDR', 3: 'dsevereDR', 4: 'eproDR'},
        "num_classes": 5,  
        "output_folder": "/home/user/yatpan/RETFound_MAE/IDRID/test_saliency_map_bicubic",
    },
    {
        "test_image_folder": "/home/user/yatpan/RETFound_MAE/APTOS2019/APTOS2019/test",
        "checkpoint_path": "/home/user/yatpan/RETFound_MAE/APTOS2019/checkpoint-best.pth",
        "class_labels": {0: 'anodr', 1: 'bmilddr', 2: 'cmoderatedr', 3: 'dseveredr', 4: 'eproliferativedr'},
        "num_classes": 5,
        "output_folder": "/home/user/yatpan/RETFound_MAE/APTOS2019/test_saliency_map",
    },
    {
        "test_image_folder": "/home/user/yatpan/RETFound_MAE/MESSIDOR2/MESSIDOR2/test",
        "checkpoint_path": "/home/user/yatpan/RETFound_MAE/MESSIDOR2/checkpoint-best.pth",
        "class_labels": {0: 'anodr', 1: 'bmilddr', 2: 'cmoderatedr', 3: 'dseveredr', 4: 'eproliferativedr'},
        "num_classes": 5,
        "output_folder": "/home/user/yatpan/RETFound_MAE/MESSIDOR2/test_saliency_map",
    },
    {
        "test_image_folder": "/home/user/yatpan/RETFound_MAE/Glaucoma_fundus/Glaucoma_fundus/test",
        "checkpoint_path": "/home/user/yatpan/RETFound_MAE/Glaucoma_fundus/checkpoint-best.pth",
        "class_labels": {0: 'anormal_control', 1: 'bearly_glaucoma', 2: 'cadvanced_glaucoma'},
        "num_classes": 3,
        "output_folder": "/home/user/yatpan/RETFound_MAE/Glaucoma_fundus/test_saliency_map",
    },
    {
        "test_image_folder": "/home/user/yatpan/RETFound_MAE/PAPILA/PAPILA/test",
        "checkpoint_path": "/home/user/yatpan/RETFound_MAE/PAPILA/checkpoint-best.pth",
        "class_labels": {0: 'anormal', 1: 'bsuspectglaucoma', 2: 'cglaucoma'},
        "num_classes": 3,
        "output_folder": "/home/user/yatpan/RETFound_MAE/PAPILA/test_saliency_map",
    }
]

# Seed setting
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = True
    print(f"Random seed set to: {seed}")

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
        model.load_state_dict(checkpoint['model'], strict=True)  # Ensures compatibility with original model
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

# We use the same way as the engine_finetune.py to calculate metrics
def misc_measures(confusion_matrix):
    acc, sensitivity, specificity, precision, G, F1_score_2, mcc_ = [], [], [], [], [], [], []

    for i in range(confusion_matrix.shape[0]):
        cm = confusion_matrix[i]
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

        # Avoid division by zero by setting default values
        acc.append((tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0)
        sensitivity_ = tp / (tp + fn) if (tp + fn) > 0 else 0
        sensitivity.append(sensitivity_)
        specificity_ = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity.append(specificity_)
        precision_ = tp / (tp + fp) if (tp + fp) > 0 else 0
        precision.append(precision_)
        G.append(math.sqrt(sensitivity_ * specificity_))
        F1_score_2.append(2 * precision_ * sensitivity_ / (precision_ + sensitivity_) if (precision_ + sensitivity_) > 0 else 0)
        mcc_.append((tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0 else 0)

    # Take the average for each metric
    metrics = {
        "Accuracy": np.mean(acc),
        "Sensitivity": np.mean(sensitivity),
        "Specificity": np.mean(specificity),
        "Precision": np.mean(precision),
        "Geometric Mean (G)": np.mean(G),
        "F1 Score": np.mean(F1_score_2),
        "MCC": np.mean(mcc_)
    }

    return metrics

# Updated calculate_metrics function to align with misc_measures
def calculate_metrics(true_labels, pred_labels, pred_probs, num_classes, output_folder):
    # Confusion matrix and ROC-AUC / Average Precision calculation
    cm = multilabel_confusion_matrix(true_labels, pred_labels, labels=list(range(num_classes)))
    metrics = misc_measures(cm)


    # Debugging print statements
    print("Debugging Information:")
    print(f"Unique classes in y_true: {np.unique(true_labels)}")
    print(f"Shape of pred_probs (y_score): {pred_probs.shape}")
    print(f"Sample of pred_probs (first 5 rows): \n{pred_probs[:5]}")
    print(f"Expected number of classes (num_classes): {num_classes}")

    auc_roc = roc_auc_score(true_labels, pred_probs, multi_class='ovr', average='macro')
    auc_pr = average_precision_score(np.array(true_labels).reshape(-1, 1) == np.arange(num_classes), pred_probs, average='macro')

    # Add AUC-ROC and AUC-PR to metrics
    metrics.update({
        "AUC-ROC": auc_roc,
        "AUC-PR": auc_pr
    })

    # Save confusion matrix image
    plt.figure(figsize=(10, 7))
    plt.imshow(cm[:, :, 0], cmap='Blues', interpolation='nearest')
    plt.colorbar()
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    cm_path = os.path.join(output_folder, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    # Save metrics to a text file
    with open(os.path.join(output_folder, "metrics.txt"), "w") as f:
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")

    return metrics

# Function to generate heatmap from LRP attributions
def show_cam_on_image(img, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

# Generate Saliency Map with Blending
def generate_visualization(original_image, attribution_generator, class_index=None, use_thresholding=False):
    # here original_image is a PIL image
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(), method="transformer_attribution", index=class_index) # use original image as input

    if transformer_attribution is None:
        raise ValueError("LRP attribution generation failed.")

    transformer_attribution = transformer_attribution.detach() # detach from computation graph to ensure no gradients are calculated

    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14) # reshape to 14x14, matching the final layer of ViT
    transformer_attribution = F.interpolate(transformer_attribution, scale_factor=16, mode='bicubic').reshape(224, 224).cpu().numpy() # upsample to 224x224 to match the input size, scale is calculated as 224/14 = 16
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())  # min-mx normalization to [0, 1]
    
    if use_thresholding:
        transformer_attribution = (transformer_attribution * 255).astype(np.uint8)
        _, transformer_attribution = cv2.threshold(transformer_attribution, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        transformer_attribution[transformer_attribution == 255] = 1

    # here the original_image is processed by transform from RETFound_MAE/util/datasets.py
    image_transformer_attribution = original_image.permute(1, 2, 0).cpu().numpy()
    image_transformer_attribution = (image_transformer_attribution - image_transformer_attribution.min()) / (image_transformer_attribution.max() - image_transformer_attribution.min())

    # Generate heatmap and blend with image
    vis = show_cam_on_image(image_transformer_attribution, transformer_attribution)
    vis = np.uint8(255 * vis)
    vis = cv2.cvtColor(np.array(vis), cv2.COLOR_RGB2BGR)
    
    return vis

# Process folder for inference and saliency maps
def process_folder(model, attribution_generator, folder_path, output_csv, class_labels, num_classes, saliency_output_folder=None, use_thresholding=False):
    # Ensure the output directory exists
    if saliency_output_folder and not os.path.exists(saliency_output_folder):
        os.makedirs(saliency_output_folder)

    true_labels, pred_labels, pred_probs = [], [], []
    results = []
    
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(root, file_name)
                true_label = next((i for i, label in class_labels.items() if label in root), None)
                print(f"true label: {true_label}")
                if true_label is None: continue

                # Inference
                pred_label, pred_prob = infer(model, img_path)
                print (f"results of inference: {pred_label}, {pred_prob}")
                true_labels.append(true_label)
                pred_labels.append(pred_label)
                pred_probs.append(pred_prob) 
                results.append([file_name, class_labels[true_label], class_labels[pred_label], pred_prob])

                # Saliency Map Generation
                input_image = transform(Image.open(img_path)).to(device)
                saliency_map = generate_visualization(input_image, attribution_generator, class_index=pred_label, use_thresholding=use_thresholding)
                
                # Modify filename for saving saliency map
                base_name = os.path.splitext(file_name)[0]  # Get filename without extension
                saliency_file_name = f"{base_name}_saliency.png"
                
                # Save the heatmap if output folder is specified
                if saliency_output_folder:
                    plt.imsave(os.path.join(saliency_output_folder, saliency_file_name), saliency_map, cmap='jet')
    
    # Debugging statements to check list lengths
    print(f"Number of true labels: {len(true_labels)}")
    print(f"Unique classes in y_true (ground truth): {np.unique(true_labels)}")
    print(f"Number of predicted labels: {len(pred_labels)}")
    print(f"Number of predicted probabilities: {len(pred_probs)}")


    metrics = calculate_metrics(true_labels, pred_labels, np.array(pred_probs), num_classes=num_classes, output_folder=saliency_output_folder)
    print(f"Metrics saved in {saliency_output_folder}")
    print(f"Metrics: {metrics}")

    # Save results to CSV
    df = pd.DataFrame(results, columns=["File Name", "True Label", "Predicted Label", "Probabilities"])
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

# Run a single experiment
def run_experiment(experiment):
    num_classes = experiment["num_classes"]
    model = load_model(
        experiment["checkpoint_path"],
        args.input_size,
        num_classes,
        args.drop_rate,
        args.global_pool
    )
    attribution_generator = LRP(model)
    
    # Run inference and save results for each experiment
    process_folder(
        model,
        attribution_generator,
        folder_path=experiment["test_image_folder"],
        output_csv=os.path.join(experiment["output_folder"], "saliency_map_predictions.csv"),
        class_labels=experiment["class_labels"],
        num_classes=num_classes,
        saliency_output_folder=experiment["output_folder"],
        use_thresholding=args.use_thresholding,
    )
    print(f"Experiment completed. Results saved in {experiment['output_folder']}")

# Main function to run
def main():
    for experiment in experiments:
        run_experiment(experiment)

if __name__ == "__main__":
    main()
