from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
from log import Log
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import numpy as np
import scipy.io as sio
import os
from my_models import ProtoNet

from LOW import LOWLoss

import matplotlib.pyplot as plt

def train_epoch_baseline(model: ProtoNet,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                epoch: int,
                device,
                log: Log = None,
                log_prefix: str = 'log_train_epochs',
                progress_prefix: str = 'Train Epoch'
                ) -> dict:

    # Store info about the procedure
    train_info = dict()
    total_loss = 0.
    total_acc = 0.

    # Create a log if required
    log_loss = f'{log_prefix}_losses'
    loss_fn = LOWLoss(lamb=0.01)

    # Make sure the model is in train mode
    model.train()

    # Show progress on progress bar
    train_iter = tqdm(enumerate(train_loader),
                      total=len(train_loader),
                      desc=progress_prefix + ' %s' % epoch,
                      ncols=0)
    # Iterate through the data set to update leaves, prototypes and network
    for i, (xs, ys) in train_iter:
        xs, ys = xs.to(device), ys.to(device)

        # Perform a forward pass through the network
        logits = model(xs)
        # loss = F.cross_entropy(logits, ys)
        loss = loss_fn(logits, ys)
        # loss += 1e0 * torch.abs(model.classifier.weight).sum(1).mean() # L1 regularization
        # W = model.W * model.mask
        # loss += 1e-1 * ((W*W).sum(0) + 1e-6).sqrt().mean()  # group lasso regularization
        ys_pred = torch.softmax(logits.detach(), dim=1)

        # Compute the gradient and update model parameters
        loss.backward()
        optimizer.step()
        model.zero_grad()

        # Count the number of correct classifications
        ys_pred_max = torch.argmax(ys_pred, dim=1)
        correct = torch.sum(torch.eq(ys_pred_max, ys))
        acc = correct.item() / float(len(xs))

        # Compute metrics over this batch
        total_loss += loss.item()
        total_acc += acc

        train_iter.set_postfix_str(
            f'Batch [{i + 1}/{len(train_loader)}], Loss: {total_loss/float(i+1):.3f}, Acc: {total_acc/float(i+1):.3f}'
        )

        if log is not None:
            log.log_values(log_loss, epoch, i + 1, loss.item(), acc)

    # plt.figure(1)
    # plt.clf()
    # plt.imshow(model.classifier.weight.detach().cpu())
    # plt.pause(.1)

    train_info['loss'] = total_loss / float(i + 1)
    train_info['train_accuracy'] = total_acc / float(i + 1)
    return train_info

@torch.no_grad()
def eval_baseline(model,
         test_loader,
         epoch,
         device,
         log: Log = None) -> dict:

    # Keep an info dict about the procedure
    info = dict()
    # Build a confusion matrix
    cm = np.zeros((model._num_classes, model._num_classes), dtype=int)
    probs = []
    lbls = []
    maps = []
    total_acc = 0.

    # Make sure the model is in evaluation mode
    model.eval()

    # Show progress on progress bar
    test_iter = tqdm(enumerate(test_loader),
                     total=len(test_loader),
                     desc='Eval Epoch %s' % epoch,
                     ncols=0)

    # Iterate through the test set
    for i, (xs, ys) in test_iter:
        xs, ys = xs.to(device), ys.to(device)

        # Use the model to classify this batch of input data
        logits = model(xs)

        loss = F.cross_entropy(logits, ys)
        ys_pred = torch.softmax(logits.detach(), dim=1)
        ys_pred_max = torch.argmax(ys_pred, dim=1)

        # Update the confusion matrix
        for y_pred, y_true in zip(ys_pred_max, ys):
            cm[y_true][y_pred] += 1
        test_iter.set_postfix_str(
            f'Batch [{i + 1}/{len(test_iter)}], Acc: {acc_from_cm(cm):.3f}'
        )
        total_acc += torch.sum(torch.eq(ys_pred_max, ys)).item() / float(len(xs))
        probs.append(ys_pred.cpu())
        lbls.append(ys.cpu())

    info['confusion_matrix'] = cm
    info['test_bal_accuracy'] = acc_from_cm(cm)
    info['test_accuracy'] = total_acc / float(i + 1)
    if log is not None and not isinstance(log, str):
        log.log_message("\nEpoch %s - Test accuracy: " % (epoch) + str(info['test_bal_accuracy']))
        log.log_message("\nEpoch %s - Test confusion matrix: " % (epoch) + str(info['confusion_matrix']))
    elif isinstance(log, str):
        savepath = os.path.join(log, 'detections')
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        sio.savemat(os.path.join(savepath, 'maps.mat'), {'maps': torch.cat(maps, 0).numpy()})

    # Save ROC AUC
    probs = torch.cat(probs, 0).numpy()
    lbls = torch.cat(lbls, 0).numpy()
    if lbls.max() > 0:
        if probs.shape[1]>2:
            clf = LogisticRegression(solver="liblinear").fit(probs, lbls)
            info['test_auc'] = 100 * roc_auc_score(lbls, clf.predict_proba(probs), multi_class='ovr')
        else:
            clf = LogisticRegression(solver="liblinear", random_state=0).fit(probs, lbls)
            info['test_auc'] = 100 * roc_auc_score(lbls, clf.predict_proba(probs)[:, 1])
    else:
        info['test_auc'] = 0.
    info['probs'] = probs

    return info


def acc_from_cm(cm: np.ndarray, balanced=True) -> float:
    """
    Compute the accuracy from the confusion matrix
    :param cm: confusion matrix
    :return: the accuracy score
    """
    assert len(cm.shape) == 2 and cm.shape[0] == cm.shape[1]

    if balanced:
        acc = 0.
        n_classes = 0
        for i in range(len(cm)):
            if np.sum(cm[i])>0:
                acc += cm[i, i] / np.sum(cm[i])
                n_classes += 1
        return acc / n_classes

    else:
        correct = 0
        for i in range(len(cm)):
            correct += cm[i, i]

        total = np.sum(cm)
        if total == 0:
            return 1
        else:
            return correct / total