import numpy as np
import matplotlib.pyplot as plt


def _plot_roc_curve(data, thresholds, title, save_path=None):
    data = np.array(data)
    data[..., 1] = (1 - data[..., 1])
    plt.plot(data[..., 1], data[..., 0], marker='o', color='#AA4499')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title(title)
    plt.xlabel("1 - Specifity")
    plt.ylabel("Sensitivity")
    plt.legend(["Thresholds from 0 to 1 in seps of 0.5"], loc="upper right", bbox_to_anchor=(0.5, -0.35, 0.5, 0.5))
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)


def generate_roc_curve(predictions, true_labels, title, save_path=None,
                       prediciton_thresholds=np.arange(0, 101, 5) / 100):
    results = list()
    tp, fp, tn, fn = 0, 0, 0, 0

    for t in prediciton_thresholds:
        preds = np.copy(predictions)
        for true_label, pred in zip(true_labels, preds):
            if pred < t:
                pred = 0
            else:
                pred = 1
            if pred == true_label:
                if pred == 1:
                    tp += 1
                else:
                    tn += 1
            else:
                if pred == 1:
                    fp += 1
                else:
                    fn += 1
        sensitivity = tp / (tp + fn)
        specifity = tn / (tn + fp)
        results.append([sensitivity, specifity])

        tp, fp, tn, fn = 0, 0, 0, 0

    _plot_roc_curve(results, prediciton_thresholds, title, save_path)