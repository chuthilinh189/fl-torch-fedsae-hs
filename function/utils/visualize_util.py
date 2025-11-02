import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

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
