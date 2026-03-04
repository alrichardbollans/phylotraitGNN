from sklearn.metrics import brier_score_loss
import torch.nn.functional as F


def test_binary(out_, data, scorer=None):
    test_predictions = out_[data.test_mask]
    probs = F.softmax(test_predictions, dim=1)  # Convert logits to probabilities.
    pred_proba = probs[:, 1]  # Probability for class 1

    pred = probs.argmax(dim=1)  # Use the class with highest probability.
    test_correct = pred == data.y[data.test_mask]  # Check against ground-truth labels.
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
    if scorer is not None:
        b_score = scorer(data.y[data.test_mask].detach().cpu().numpy(), pred_proba.detach().cpu().numpy())
    else:
        # To use brier_score_loss:
        b_score = brier_score_loss(
            data.y[data.test_mask].detach().cpu().numpy(),
            pred_proba.detach().cpu().numpy()
        )
    return test_acc, b_score


if __name__ == '__main__':
    main()
