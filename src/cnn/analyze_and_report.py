import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report

def analyze_and_report(cnn_model, gnn_features, ideal_values, condition_label):
    """
    Analyze the GNN features using the CNN and compare with ideal FMS profile.
    :param cnn_model: Trained CNN model.
    :param gnn_features: Processed features from GNN.
    :param ideal_values: Ideal metrics for FMS.
    :param condition_label: 'fibro' or 'normal', as indicated by GAN.
    """
    print("Analyzing and reporting...")
    cnn_model.eval()
    with torch.no_grad():
        cnn_output = cnn_model(gnn_features)

    # Convert ideal_values to a tensor
    ideal_tensor = torch.tensor(ideal_values, dtype=torch.float).unsqueeze(0)

    # Ensure dimensions are compatible for cosine similarity
    if cnn_output.shape[1] != ideal_tensor.shape[1]:
        # Adjust dimensions of ideal_tensor to match cnn_output
        ideal_tensor = torch.nn.functional.pad(ideal_tensor, (0, cnn_output.shape[1] - ideal_tensor.shape[1]), "constant", 0)

    similarity = cosine_similarity(cnn_output.detach().numpy(), ideal_tensor.detach().numpy())[0][0] * 100
    print(f"Similarity: {similarity}%")

    # Adjust threshold dynamically based on ROC-AUC
    y_true = [1 if condition_label == 'Fibro' else 0]
    y_scores = [similarity]
    unique_labels, counts = np.unique(y_true, return_counts=True)
    print("Labels distribution:", dict(zip(unique_labels, counts)))

    if len(unique_labels) > 1:
        roc_auc = roc_auc_score(y_true, y_scores)
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        optimal_threshold = thresholds[np.argmax(precision * recall)]
        print("Optimal Threshold:", optimal_threshold)
    else:
        roc_auc = None
        optimal_threshold = 0.5  # Default threshold

    if similarity > optimal_threshold:
        prediction = 'Fibro'
    else:
        prediction = 'Normal'

    print(f"Condition: {condition_label}, Prediction: {prediction}, Similarity: {similarity}%, ROC-AUC: {roc_auc}")

    # Print classification report
    print(classification_report(y_true, [1 if similarity > optimal_threshold else 0]))