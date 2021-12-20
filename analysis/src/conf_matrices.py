import sklearn.metrics as skm
import numpy as np
import matplotlib.pyplot as plt
import operator
import itertools


def _plot_confusion_matrix(cm,
                           classes,
                           normalize=False,
                           title='Confusion matrix',
                           cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=8)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=8)
    plt.yticks(tick_marks, classes, fontsize=8)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label', fontsize=8)
    plt.xlabel('Predicted label', fontsize=8)
    plt.tight_layout()


def generate_confusion_matrix(predictions, true_labels, title):
    labels = ['C', 'DS', 'TC', 'N']

    y_pred = list()
    y_true = list()
    for true, pred in zip(true_labels, predictions):
        max_idx, _ = max(enumerate(pred), key=operator.itemgetter(1))
        y_pred.append(labels[max_idx])
        y_true.append(labels[true.tolist().index(1)])

    conf_mat = skm.confusion_matrix(np.array(y_true), np.array(y_pred), labels=labels)

    acc = skm.accuracy_score(np.array(y_true), np.array(y_pred))
    print(acc)
    _plot_confusion_matrix(conf_mat, classes=labels,  title=title)

