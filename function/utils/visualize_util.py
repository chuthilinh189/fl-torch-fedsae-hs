import os
import csv
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import roc_auc_score
import torch
from sklearn.preprocessing import StandardScaler
try:
    import umap
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

def visualize_parameters(epoch, clients, poisoned_workers, args, pdf_writer):
    client_weights = []
    client_labels = []

    for i, client in enumerate(clients):
        weights = client.get_nn_parameters()
        flat_weights = np.concatenate([w.flatten() for w in weights.values()])
        client_weights.append(flat_weights)
        client_labels.append(1 if i in poisoned_workers else 0)

    client_weights = np.array(client_weights)

    perplexity = min(len(client_weights) - 1, 5)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    reduced_weights = tsne.fit_transform(client_weights)

    plt.figure(figsize=(8, 8))
    for i, label in enumerate(client_labels):
        color = 'red' if label == 1 else 'blue'
        marker = 'x' if label == 1 else 'o'
        plt.scatter(reduced_weights[i, 0], reduced_weights[i, 1], c=color, marker=marker)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())

    plt.title(f"t-SNE Visualization of Client Weights (Epoch {epoch})")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.grid(True)

    pdf_writer.savefig()
    plt.close()


def log_re_distributions_from_lists(list_re, raw_labels, out_dir, epoch, client_id=None, normalize_per_client=False):
    """
    Save RE per-sample grouped by attack type and draw simple plots (histogram, boxplot).

    Returns (summary_dict, grouped_dict)
    summary_dict: attack_id -> {count, mean, std}
    grouped_dict: attack_id -> list of RE floats
    """
    os.makedirs(out_dir, exist_ok=True)

    re_arr = np.array(list_re, dtype=float)
    labels_arr = np.array(raw_labels, dtype=int)

    if normalize_per_client:
        mu, sigma = re_arr.mean(), re_arr.std() if re_arr.std() > 0 else 1.0
        re_arr = (re_arr - mu) / sigma

    grouped = defaultdict(list)
    for r, l in zip(re_arr.tolist(), labels_arr.tolist()):
        grouped[l].append(r)

    summary = {}
    fname_base = f"epoch{epoch}"
    if client_id is not None:
        fname_base += f"_client{client_id}"

    # write per-attack CSVs and create simple plots
    for atk, rlist in grouped.items():
        rarr = np.array(rlist)
        summary[atk] = {"count": int(len(rarr)), "mean": float(rarr.mean()), "std": float(rarr.std())}

        csv_path = os.path.join(out_dir, f"{fname_base}_attack{atk}_re.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["re"])
            for v in rarr:
                writer.writerow([v])

        # histogram
        plt.figure(figsize=(6, 4))
        plt.hist(rarr, bins=40, color="C0", alpha=0.8)
        plt.title(f"RE histogram - attack {atk} | epoch {epoch} | cnt={len(rarr)}")
        plt.xlabel("Reconstruction error")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{fname_base}_attack{atk}_hist.png"))
        plt.close()

        # boxplot (single)
        plt.figure(figsize=(3, 4))
        plt.boxplot(rarr, vert=False)
        plt.title(f"RE box - attack {atk}")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{fname_base}_attack{atk}_box.png"))
        plt.close()

    # overall box across attacks
    if len(grouped) > 1:
        atk_ids = sorted(grouped.keys())
        data = [grouped[a] for a in atk_ids]
        plt.figure(figsize=(max(6, len(atk_ids) * 0.6), 4))
        plt.boxplot(data)
        plt.xticks(range(1, len(atk_ids) + 1), [str(a) for a in atk_ids])
        plt.title(f"RE per attack type - epoch {epoch}")
        plt.xlabel("attack id")
        plt.ylabel("Reconstruction error")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{fname_base}_per_attack_box.png"))
        plt.close()

    return summary, grouped


def collect_re_and_labels_from_client(client, normalize=False):
    """
    Collect per-sample reconstruction error and raw labels from a client's test_data_loader.

    Returns (list_re, raw_labels)
    """
    client.net.eval()
    list_re = []
    raw_labels = []
    device = client.device
    with torch.no_grad():
        for input, label in client.test_data_loader:
            input = input.to(device)
            label = label.to(device)
            encode, decode = client.net(input)
            # per-sample MSE
            per_sample_re = ((decode - input) ** 2).mean(dim=1).cpu().numpy()
            list_re.extend(per_sample_re.tolist())
            raw_labels.extend(label.cpu().numpy().astype(int).tolist())

    if normalize and len(list_re) > 0:
        arr = np.array(list_re)
        arr = (arr - arr.mean()) / (arr.std() if arr.std() > 0 else 1)
        list_re = arr.tolist()

    return list_re, raw_labels


def collect_z_norms_and_labels_from_client(client, normalize=False):
    """
    Collect per-sample latent vector norms (L2) and raw labels from a client's test_data_loader.

    Returns (list_z_norms, raw_labels)
    """
    client.net.eval()
    list_z = []
    raw_labels = []
    device = client.device
    with torch.no_grad():
        for input, label in client.test_data_loader:
            input = input.to(device)
            label = label.to(device)
            encode, decode = client.net(input)
            # per-sample latent L2 norm
            per_sample_z_norm = torch.norm(encode.view(encode.size(0), -1), p=2, dim=1).cpu().numpy()
            list_z.extend(per_sample_z_norm.tolist())
            raw_labels.extend(label.cpu().numpy().astype(int).tolist())

    if normalize and len(list_z) > 0:
        arr = np.array(list_z)
        arr = (arr - arr.mean()) / (arr.std() if arr.std() > 0 else 1)
        list_z = arr.tolist()

    return list_z, raw_labels


def log_z_boxplots_from_lists(list_z, raw_labels, out_dir, epoch, client_id=None):
    """
    For each attack type, create a combined boxplot comparing benign (label 0) vs that attack's latent-norm distribution.
    Also write per-attack CSVs of z-norms. Produces per-client files when client_id is provided, and global when None.
    """
    os.makedirs(out_dir, exist_ok=True)

    z_arr = np.array(list_z, dtype=float)
    labels_arr = np.array(raw_labels, dtype=int)

    fname_base = f"epoch{epoch}"
    if client_id is not None:
        fname_base += f"_client{client_id}"

    # group by label
    grouped = defaultdict(list)
    for z, l in zip(z_arr.tolist(), labels_arr.tolist()):
        grouped[l].append(z)

    # write per-label CSVs
    for lbl, zlist in grouped.items():
        csv_path = os.path.join(out_dir, f"{fname_base}_label{lbl}_z_norm.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["z_norm"])
            for v in zlist:
                writer.writerow([v])

    # For each attack (label != 0) create a combined boxplot (benign vs attack)
    ben_list = grouped.get(0, [])
    for atk, atk_list in grouped.items():
        if atk == 0:
            continue

        data = [ben_list, atk_list]
        plt.figure(figsize=(6, 4))
        plt.boxplot(data)
        plt.xticks([1, 2], ["benign", f"attack_{atk}"])
        plt.title(f"Latent L2-norm: benign vs attack {atk} | epoch {epoch}")
        plt.ylabel("L2 norm of latent z")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{fname_base}_attack{atk}_z_box_compare.png"))
        plt.close()

    # overall per-attack box (for visualization across attacks show distribution of z-norm per attack)
    atk_ids = sorted([k for k in grouped.keys()])
    if len(atk_ids) > 1:
        data = [grouped[a] for a in atk_ids]
        plt.figure(figsize=(max(6, len(atk_ids) * 0.6), 4))
        plt.boxplot(data)
        plt.xticks(range(1, len(atk_ids) + 1), [str(a) for a in atk_ids])
        plt.title(f"Latent L2-norm per attack type - epoch {epoch}")
        plt.xlabel("attack id")
        plt.ylabel("L2 norm of latent z")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"{fname_base}_per_attack_z_box.png"))
        plt.close()

    return grouped


def compute_auc_per_attack_from_flat(y_true, re_scores):
    """
    Compute one-vs-rest AUC per attack id using benign (label 0) as negatives.
    Returns dict attack_id -> auc (float) or None if not computable.
    """
    y_true = np.array(y_true)
    re_scores = np.array(re_scores)
    unique_labels = sorted(np.unique(y_true))
    aucs = {}
    for atk in unique_labels:
        if atk == 0:
            continue
        pos_mask = (y_true == atk)
        neg_mask = (y_true == 0)
        if neg_mask.sum() == 0 or pos_mask.sum() == 0:
            aucs[atk] = None
            continue
        y_bin = np.concatenate([np.ones(pos_mask.sum()), np.zeros(neg_mask.sum())])
        scores = np.concatenate([re_scores[pos_mask], re_scores[neg_mask]])
        try:
            aucs[atk] = float(roc_auc_score(y_bin, scores))
        except Exception:
            aucs[atk] = None
    return aucs


def collect_latents_and_labels_from_client(client, max_samples_per_class=None):
    """
    Collect per-sample latent vectors and raw labels from a client's test_data_loader.

    Returns (latents_numpy_array, labels_list) where latents shape = (N, D).
    If max_samples_per_class is provided, this will sample up to that many examples per label to keep embeddings fast.
    """
    client.net.eval()
    latents = []
    labels = []
    device = client.device
    with torch.no_grad():
        for input, label in client.test_data_loader:
            input = input.to(device)
            label = label.to(device)
            encode, decode = client.net(input)
            enc_np = encode.view(encode.size(0), -1).cpu().numpy()
            latents.append(enc_np)
            labels.extend(label.cpu().numpy().astype(int).tolist())

    if len(latents) == 0:
        return np.zeros((0, 0)), []

    latents = np.concatenate(latents, axis=0)

    # optional balanced sampling per class to limit size
    if max_samples_per_class is not None and max_samples_per_class > 0:
        kept_idx = []
        labels_arr = np.array(labels)
        for lbl in np.unique(labels_arr):
            idxs = np.where(labels_arr == lbl)[0]
            if len(idxs) > max_samples_per_class:
                chosen = np.random.choice(idxs, size=max_samples_per_class, replace=False)
            else:
                chosen = idxs
            kept_idx.extend(chosen.tolist())
        kept_idx = sorted(kept_idx)
        latents = latents[kept_idx]
        labels = labels_arr[kept_idx].tolist()

    return latents, labels


def plot_latent_embedding(latents, labels, out_dir, epoch, client_id=None, proto_z0=None, proto_z1=None, method="tsne", max_points=2000, random_state=42):
    """
    Create a 2D scatter of latent vectors using t-SNE or UMAP, color by label and overlay prototypes when provided.

    - latents: numpy array shape (N, D)
    - labels: list/array shape (N,)
    - out_dir: directory to save images
    - epoch, client_id: used for filename
    - proto_z0/proto_z1: numpy arrays of same latent dimension or None
    - method: 'tsne' or 'umap' (umap requires umap-learn installed)
    """
    os.makedirs(out_dir, exist_ok=True)
    if latents is None or latents.shape[0] == 0:
        return

    N = latents.shape[0]
    labels_arr = np.array(labels)

    # sample if too many points
    if N > max_points:
        # sample balanced by label
        kept = []
        for lbl in np.unique(labels_arr):
            idxs = np.where(labels_arr == lbl)[0]
            k = max_points // max(1, len(np.unique(labels_arr)))
            if len(idxs) > k:
                chosen = np.random.choice(idxs, size=k, replace=False).tolist()
            else:
                chosen = idxs.tolist()
            kept.extend(chosen)
        kept = sorted(kept)
        latents_plot = latents[kept]
        labels_plot = labels_arr[kept]
    else:
        latents_plot = latents
        labels_plot = labels_arr

    # standardize
    try:
        scaler = StandardScaler()
        latents_plot = scaler.fit_transform(latents_plot)
    except Exception:
        pass

    # include prototypes in the embedding transform if present so they are mapped consistently
    proto_stack = []
    proto_labels = []
    if proto_z0 is not None:
        proto_stack.append(proto_z0.reshape(1, -1))
        proto_labels.append(-1)
    if proto_z1 is not None:
        proto_stack.append(proto_z1.reshape(1, -1))
        proto_labels.append(-2)

    try:
        if method == "umap" and _HAS_UMAP:
            reducer = umap.UMAP(n_components=2, random_state=random_state)
        else:
            reducer = TSNE(n_components=2, init='pca', random_state=random_state)

        if len(proto_stack) > 0:
            combined = np.vstack([latents_plot] + proto_stack)
            emb = reducer.fit_transform(combined)
            emb_points = emb[: latents_plot.shape[0]]
            emb_protos = emb[latents_plot.shape[0]:]
        else:
            emb_points = reducer.fit_transform(latents_plot)
            emb_protos = None
    except Exception as e:
        # fallback to PCA if t-SNE/umap fails
        try:
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            emb_points = pca.fit_transform(latents_plot)
            emb_protos = None
        except Exception:
            return

    fname_base = f"epoch{epoch}"
    if client_id is not None:
        fname_base += f"_client{client_id}"

    plt.figure(figsize=(7, 6))
    unique_lbls = sorted(np.unique(labels_plot))
    colors = plt.cm.tab10(np.linspace(0, 1, max(10, len(unique_lbls))))
    for i, lbl in enumerate(unique_lbls):
        mask = labels_plot == lbl
        plt.scatter(emb_points[mask, 0], emb_points[mask, 1], s=8, c=[colors[i]], label=str(lbl), alpha=0.7)

    # overlay prototypes
    if emb_protos is not None:
        for pidx, plabel in enumerate(proto_labels):
            px, py = emb_protos[pidx]
            if plabel == -1:
                plt.scatter(px, py, c='green', marker='X', s=120, edgecolor='k', label='proto_z0')
            elif plabel == -2:
                plt.scatter(px, py, c='red', marker='X', s=120, edgecolor='k', label='proto_z1')

    plt.legend(markerscale=2)
    plt.title(f"Latent embedding ({method}) - epoch {epoch}" + (f" client {client_id}" if client_id is not None else ""))
    plt.xlabel("dim1")
    plt.ylabel("dim2")
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{fname_base}_{method}.png")
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path
