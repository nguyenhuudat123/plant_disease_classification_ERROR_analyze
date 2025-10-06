import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def save_misclassified_images(dataset, preds, targets, probs, classes, save_dir, n=20):
    """Lưu một số ảnh bị dự đoán sai kèm nhãn thật, nhãn dự đoán và confidence."""
    os.makedirs(save_dir, exist_ok=True)
    preds = np.array(preds)
    targets = np.array(targets)
    probs = np.array(probs)

    mis_idx = [i for i, (p, t) in enumerate(zip(preds, targets)) if p != t]
    random.shuffle(mis_idx)

    for k, idx in enumerate(mis_idx[:n]):
        img, _ = dataset[idx]
        img = img.permute(1, 2, 0).cpu().numpy()
        plt.imshow(img)
        plt.title(f"True: {classes[targets[idx]]}, Pred: {classes[preds[idx]]}, Conf: {probs[idx].max():.2f}")
        plt.axis("off")
        plt.savefig(os.path.join(save_dir, f"misclassified_{k}.png"))
        plt.close()

    print(f"Saved {min(n, len(mis_idx))} misclassified images to {save_dir}")


def compute_topk_accuracy(probs, targets, ks=(1, 3, 5)):
    """Tính top-k accuracy cho các k trong ks."""
    probs = torch.tensor(probs)
    targets = torch.tensor(targets)
    max_k = max(ks)
    _, pred = probs.topk(max_k, dim=1, largest=True, sorted=True)

    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    results = {}
    for k in ks:
        acc = correct[:, :k].sum().item() / len(targets) * 100
        results[f"top_{k}_acc"] = acc
    return results


def plot_classwise_comparison(stats_files, save_path="classwise_comparison.png"):
    """
    Vẽ so sánh F1-score của nhiều model từ các file classification_stats.csv.
    stats_files: dict {model_name: path_to_csv}
    """
    data = []
    for model_name, path in stats_files.items():
        df = pd.read_csv(path)
        df["Model"] = model_name
        data.append(df[["Class", "F1_Score", "Model"]])

    df_all = pd.concat(data, axis=0)

    pivot = df_all.pivot(index="Class", columns="Model", values="F1_Score")
    pivot.plot(kind="bar", figsize=(12, 6))
    plt.ylabel("F1-score")
    plt.title("Per-Class F1-score Comparison")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Classwise comparison plot saved to {save_path}")

