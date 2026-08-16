import torch


class EarlyStopping:
    # Modified from: https://www.geeksforgeeks.org/deep-learning/how-to-handle-overfitting-in-pytorch-models-using-early-stopping/
    def __init__(self, patience=5, delta=0, epoch_minimum=10):
        self.patience = patience
        self.delta = delta  # Minimum change in the monitored quantity to qualify as an improvement
        self.best_score = None
        self.early_stop = False
        self.counter = 0
        self.total_counter = 0
        self.epoch_minimum = epoch_minimum
        self.best_model_state = None

    def __call__(self, val_loss, model):
        score = -val_loss
        self.total_counter += 1

        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
        elif score < self.best_score + self.delta:
            self.counter += 1
            if (self.counter >= self.patience) and self.total_counter >= self.epoch_minimum:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

    def load_best_model(self, model):
        model.load_state_dict(self.best_model_state)


def train_gcn_model(model, data, loss_function, optimizer, epochs, plot_loss=False, early_stopping=None):
    if early_stopping is not None and not (hasattr(data, 'val_mask')):
        raise ValueError('Early stopping is only supported for datasets with a validation mask.')

    train_losses = []
    val_losses = []
    for epoch in range(1, epochs):
        train_loss, val_loss = model.train_step(data, optimizer, loss_function)
        train_losses.append(train_loss.detach().cpu().numpy())
        if val_loss is not None:
            val_losses.append(val_loss.detach().cpu().numpy())
            if early_stopping is not None:
                early_stopping(val_loss, model)
                if early_stopping.early_stop:
                    # print("Early stopping")
                    break
    if plot_loss:
        import matplotlib.pyplot as plt
        plt.plot(train_losses, label='train')
        if len(val_losses) > 0:
            plt.plot(val_losses, label='val')
        plt.legend()
        plt.show()
    if early_stopping is not None:
        early_stopping.load_best_model(model)
