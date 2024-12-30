import torch
from sklearn.metrics.pairwise import cosine_similarity

def analyze_and_report(cnn_model, gnn_features, ideal_fms_profile, condition_label):
    """
    Analyze the GNN features using the CNN and compare with ideal FMS profile.
    :param cnn_model: Trained CNN model.
    :param gnn_features: Processed features from GNN.
    :param ideal_fms_profile: Ideal metrics for FMS.
    :param condition_label: 'fibro' or 'normal', as indicated by GAN.
    """
    # Pass features through CNN
    cnn_model.eval()
    with torch.no_grad():
        cnn_output = cnn_model(gnn_features)

    # Calculate similarity to ideal FMS profile
    ideal_tensor = torch.tensor(ideal_fms_profile, dtype=torch.float).unsqueeze(0)
    similarity = cosine_similarity(cnn_output.numpy(), ideal_tensor.numpy())[0][0] * 100

    # Report findings
    print(f"GAN Label: {condition_label}")
    print(f"Similarity to Ideal FMS Profile: {similarity:.2f}%")
    print(f"Detected Condition: {'FMS' if similarity > 75 else 'Normal'}\n")  # Threshold for FMS