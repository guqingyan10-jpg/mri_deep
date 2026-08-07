"""
Evaluation metrics for BraTS2020 Model Assessment.
Extracted from: MultiModel XAI Brats2020.ipynb (cells 70, 71, 72, 84, 98, 116)

Contains:
  - compute_metrics: TP/FP/TN/FN pixel-level metrics
  - metric: accuracy, precision, recall, f1_score
  - plot_confusion_matrix
  - compute_scores_per_classes / compute_scores_per_classes_mean
  - compute_results: collect predictions for visualization
"""

import gc
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

from training.metrics import dice_coef_metric_per_classes, jaccard_coef_metric_per_classes


# ============================================================
# Pixel-level Metrics (cell 70)
# ============================================================

def compute_metrics(model, dataloader, threshold=0.33):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    true_positives = 0
    false_positives = 0
    true_negatives = 0
    false_negatives = 0

    counter = 0  # Counter to keep track of the number of entries processed

    with torch.no_grad():  # Disable gradient calculations to save memory
        for data in dataloader:

            images, targets = data['image'], data['mask']
            images = images.to(device)
            targets = targets.to(device)

            logits = model(images)
            if isinstance(logits, tuple):
                logits = logits[0]
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= threshold).float()

            # Compute binary segmentation metrics
            true_positives += torch.sum((predictions == 1) & (targets == 1)).item()
            false_positives += torch.sum((predictions == 1) & (targets == 0)).item()
            true_negatives += torch.sum((predictions == 0) & (targets == 0)).item()
            false_negatives += torch.sum((predictions == 0) & (targets == 1)).item()

            counter += 1

            # Free memory by clearing intermediate variables
            del images, targets, logits, probabilities, predictions
            torch.cuda.empty_cache()

    return true_positives , false_positives , true_negatives , false_negatives


# ============================================================
# Derived Metrics (cell 72)
# ============================================================

def metric(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)
    return accuracy, precision, recall, f1_score


# ============================================================
# Confusion Matrix Plot (cell 71)
# ============================================================

def plot_confusion_matrix(ax, tp, fp, tn, fn, title):
    # Create confusion matrix array
    confusion_matrix = np.array([[tp, fp], [fn, tn]])

    # Set up labels for matrix
    labels = ['True ', 'False ']

    # Create color map
    cmap = plt.cm.Blues

    # Plot confusion matrix
    cax = ax.matshow(confusion_matrix, interpolation='nearest', cmap=cmap)
    ax.set_title(title)

    # Add colorbar to the figure
    fig.colorbar(cax, ax=ax)

    # Add labels to matrix cells
    thresh = confusion_matrix.max() / 2.
    for i, j in np.ndindex(confusion_matrix.shape):
        ax.text(j, i, format(confusion_matrix[i, j], 'd'), horizontalalignment='center', color='white' if confusion_matrix[i, j] > thresh else 'black')

    # Set tick labels
    tick_marks = np.arange(len(labels))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(labels, rotation=45)
    ax.set_yticklabels(labels)

    # Set axis labels
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')


# ============================================================
# Per-Class Score Computation (cell 98)
# ============================================================

def compute_scores_per_classes(model,          # model
                               dataloader,     # tuple consisting of ( id , image tensor , mask tensor )
                               classes):       # classes : WT , TC , ET
    """
    Compute Dice and Jaccard coefficients for each class.
    Params:
        model: neural net for make predictions.
        dataloader: dataset object to load data from.
        classes: list with classes.
        Returns: dictionaries with dice and jaccard coefficients for each class for each slice.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dice_scores_per_classes = {key: list() for key in classes}
    iou_scores_per_classes = {key: list() for key in classes}
    haus_scores_per_classes = {key: list() for key in classes}

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            imgs, targets = data['image'], data['mask']
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            if isinstance(logits, tuple):
                logits = logits[0]
            logits = logits.detach().cpu().numpy()
            targets = targets.detach().cpu().numpy()

            # Now finding the overlap between the raw prediction i.e. logit & the mask i.e. target & finding the dice & iou scores
            dice_scores = dice_coef_metric_per_classes(logits, targets)
            iou_scores = jaccard_coef_metric_per_classes(logits, targets)
            #haus_scores = hausdorff_distance_metric_per_class(logits, targets)

            # storing both dice & iou scores in the list declared
            for key in dice_scores.keys():
                dice_scores_per_classes[key].extend(dice_scores[key])

            for key in iou_scores.keys():
                iou_scores_per_classes[key].extend(iou_scores[key])

            # for key in iou_scores.keys():
            #     haus_scores_per_classes[key].extend(haus_scores[key])

    return dice_scores_per_classes, iou_scores_per_classes


def compute_scores_per_classes_mean(model,
                               dataloader,
                               classes):

    dice_scores_per_classes, iou_scores_per_classes = compute_scores_per_classes(model,
                               dataloader,
                               classes)


    dice_means = {key: np.mean(values) for key, values in dice_scores_per_classes.items()}
    iou_means = {key: np.mean(values) for key, values in iou_scores_per_classes.items()}

    return dice_means, iou_means


# ============================================================
# Prediction Collection (cell 116)
# ============================================================

def compute_results(model,
                    dataloader,
                    treshold=0.33):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = {"Id": [],"image": [], "GT": [],"Prediction": []}

    with torch.no_grad():
        for i, data in enumerate(dataloader):
            id_, imgs, targets = data['Id'], data['image'], data['mask']
            imgs, targets = imgs.to(device), targets.to(device)
            logits = model(imgs)
            if isinstance(logits, tuple):
                logits = logits[0]

            probs = torch.sigmoid(logits)

            predictions = (probs >= treshold).float()
            predictions =  predictions.cpu()
            targets = targets.cpu()

            results["Id"].append(id_)
            results["image"].append(imgs.cpu())
            results["GT"].append(targets)
            results["Prediction"].append(predictions)

            # only 5 pars
            if (i > 5):
                return results
        return results


# ============================================================
# Print formatted metrics table
# ============================================================

def print_metrics_table(dice_means, iou_means, model_name="Model"):
    """Print a formatted table of per-class metrics."""
    print(f"\n{'='*60}")
    print(f"  {model_name} — Evaluation Results")
    print(f"{'='*60}")
    print(f"{'Class':<10} {'Dice Score':<15} {'IoU/Jaccard':<15}")
    print(f"{'-'*40}")
    for cls in dice_means.keys():
        print(f"{cls:<10} {dice_means[cls]:<15.4f} {iou_means[cls]:<15.4f}")
    print(f"{'-'*40}")
    avg_dice = np.mean(list(dice_means.values()))
    avg_iou = np.mean(list(iou_means.values()))
    print(f"{'Average':<10} {avg_dice:<15.4f} {avg_iou:<15.4f}")
    print(f"{'='*60}\n")
    return avg_dice, avg_iou
