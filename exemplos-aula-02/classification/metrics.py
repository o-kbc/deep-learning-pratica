
from sklearn import metrics


def get_scores(targets, predictions):
    return {
        "accuracy": metrics.accuracy_score(targets, predictions),
        "balanced_accuracy": metrics.balanced_accuracy_score(targets, predictions),
        "precision": metrics.precision_score(targets, predictions, average="weighted"),
        "recall": metrics.recall_score(targets, predictions, average="weighted"),
        "f1_score": metrics.f1_score(targets, predictions, average="weighted")
    }


def accuracy_per_class(targets, predictions, labels=None):
    cmatrix = metrics.confusion_matrix(targets, predictions, labels=labels)
    cmatrix = cmatrix / (cmatrix.astype(float).sum(axis=1) + 1E-8)
    accuracies = cmatrix.diagonal()
    return accuracies


def validate(loss, preds, targets):   
    scores = get_scores(targets, preds)
    classes_scores = accuracy_per_class(targets, preds,
                                                labels=list(range(10)))
    return scores, classes_scores