""""This module contains an evaluation procedure for video anomaly detection."""

import argparse
import os
from os import path
import torch
from torch.backends import cudnn

from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

from network.TorchUtils import TorchModel
from features_loader import FeaturesLoaderVal


def get_args() -> argparse.Namespace:
    """Reads command line args and returns the parser object the represent the specified arguments."""
    parser = argparse.ArgumentParser(
        description="Video Anomaly Detection Evaluation Parser"
    )
    parser.add_argument(
        "--features_path", default="features_output", help="path to features"
    )
    parser.add_argument(
        "--feature_dim", type=int, default=4096, help="feature dimension"
    )
    parser.add_argument(
        "--annotation_path", default="Test_Annotation.txt", help="path to annotations"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./exps/c3d/models/epoch_80000.pt",
        help="set logging file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = FeaturesLoaderVal(
        features_path=args.features_path,
        feature_dim=args.feature_dim,
        annotation_path=args.annotation_path,
    )

    data_iter = torch.utils.data.DataLoader(
        data_loader,
        batch_size=1,
        shuffle=False,
        num_workers=0,  # 4, # change this part accordingly
        pin_memory=True,
    )

    model = TorchModel.load_model(args.model_path).to(device).eval()

    # enable cudnn tune
    cudnn.benchmark = True
    anomaly_score_folder = "anomaly_scores"
    if not path.exists(anomaly_score_folder):
            os.mkdir(anomaly_score_folder)

    x = [i for i in range(32)]
    with torch.no_grad():
        count = 0
        for features, start_end_couples, lengths in tqdm(data_iter):
            # features is a batch where each item is a tensor of 32 4096D features
            features = features.to(device)
            outputs = model(features).squeeze(-1)  # (batch_size, 32)
            y = np.array(outputs)
            plt.clf()
            plt.plot(x, y[0])
            plt.savefig(path.join(anomaly_score_folder, str(count)))
            count += 1