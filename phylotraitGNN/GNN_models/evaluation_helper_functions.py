from sklearn.metrics import brier_score_loss
import torch.nn.functional as F


def convert_logits_to_probs(logits):
    # check that test predictions are logits, not already softmaxed
    assert not (logits.sum(dim=1) == 1).all()

    probs = F.softmax(logits, dim=1)  # Convert logits to probabilities.
    # assert the row sums are 1
    assert (probs.sum(dim=1) < 1.001).all()
    assert (probs.sum(dim=1) > 0.999).all()
    return probs


def test_binary_probs(probs, data, mask, scorer=None):
    pred_proba = probs[:, 1]  # Probability for class 1

    pred = probs.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred == data.y[mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
    if scorer is not None:
        b_score = scorer(data.y[mask].detach().cpu().numpy(), pred_proba.detach().cpu().numpy())
    else:
        # To use brier_score_loss:
        b_score = brier_score_loss(
            data.y[mask].detach().cpu().numpy(),
            pred_proba.detach().cpu().numpy()
        )
    return test_acc, b_score


def test_binary_GNN_outputs(out_, data, mask, scorer=None):
    # For testing use mask = data.test_mask
    # For validation use mask = data.val_mask

    test_predictions = out_[mask]

    probs = convert_logits_to_probs(test_predictions)

    return test_binary_probs(probs, data, mask, scorer)


if __name__ == '__main__':
    main()
