import torch
from sklearn.metrics import brier_score_loss, mean_absolute_error
import torch.nn.functional as F


def convert_logits_to_probs(logits):
    # check that test predictions are logits, not already softmaxed
    assert not (logits.sum(dim=1) == 1).all()

    probs = F.softmax(logits, dim=1)  # Convert logits to probabilities.
    # assert the row sums are 1
    assert (probs.sum(dim=1) < 1.001).all()
    assert (probs.sum(dim=1) > 0.999).all()
    return probs


def test_binary_probs(probs, data, mask, scorer):
    pred_proba = probs[mask][:, 1]  # Probability for class 1

    # pred = probs.argmax(dim=1)  # Use the class with highest probability.
    # test_correct = pred == data.y[mask]  # Check against ground-truth labels.
    # test_acc = int(test_correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.

    b_score = scorer(data.y[mask].detach().cpu().numpy(), pred_proba.detach().cpu().numpy())

    return b_score


def test_binary_GNN_outputs(out_, data, mask, scorer):
    # For testing use mask = data.test_mask
    # For validation use mask = data.val_mask

    probs = convert_logits_to_probs(out_)

    return test_binary_probs(probs, data, mask, scorer)


def check_regression_output(preds):
    assert preds.shape[-1] == 1 or preds.dim() == 1
    assert torch.isfinite(preds).all()  # no NaN/Inf
    return preds


def test_regression_GNN_outputs(out_, data, mask, scorer: callable):
    # For testing use mask = data.test_mask
    # For validation use mask = data.val_mask

    check_regression_output(out_)

    _score = scorer(data.y[mask].detach().cpu().numpy(), out_[mask].detach().cpu().numpy())
    return _score
