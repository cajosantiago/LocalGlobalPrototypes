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
    total_acc_local = 0.
    total_acc_global = 0.
    local_counter = torch.zeros((model._num_classes, model.npl))
    global_counter = torch.zeros((model._num_classes, model.npg))
    if model.local_prototypes:
        local_clusters = model.prototype_layer_local.prototype_vectors.clone()
    if model.global_prototypes:
        global_clusters = model.prototype_layer_global.prototype_vectors.clone()

    # Create a log if required
    log_loss = f'{log_prefix}_losses'

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
        xg, xl, feat = model(xs, get_features=True)

        loss = 0.
        ys_pred_global = 0.
        ys_pred_local = 0.
        if not model.global_prototypes and not model.local_prototypes: # standard classifier
            loss += F.cross_entropy(xg, ys)
            ys_pred_global = torch.softmax(xg.detach(), dim=1)
        else:
            if torch.is_tensor(xg):                                    # global prototypes
                xg_r = xg.view(-1, model._num_classes, model.npg)
                best_perclass, indices = F.adaptive_max_pool1d(xg_r, 1, return_indices=True)
                best_perclass = torch.flatten(best_perclass, start_dim=1)
                best_perclass = torch.flatten(F.adaptive_avg_pool1d(xg_r, 1), start_dim=1)

                with torch.no_grad():
                    for j, (lbl, closest) in enumerate(zip(ys, indices)):
                        global_counter[lbl, closest[lbl]] += 1
                        eta = (1. / global_counter[lbl, closest[lbl]]).to(device)
                        idx = model.npg * lbl + closest[lbl]
                        global_clusters[idx] = (1-eta)*global_clusters[idx] + eta*feat[1][j:j+1]

                loss += F.cross_entropy(best_perclass, ys)
                loss += 1e-2 * F.pairwise_distance(model.prototype_layer_global.prototype_vectors.squeeze(), global_clusters.squeeze()).mean()
                ys_pred_global = torch.softmax(best_perclass.detach(), dim=1)

            if torch.is_tensor(xl):                                    # local prototypes
                bestdistance, patch_idx = F.adaptive_max_pool2d(xl, (1,1), return_indices=True)
                bestdistance_r = bestdistance.squeeze().view(-1, model._num_classes, model.npl)

                bestdistance_perclass, indices = F.adaptive_max_pool1d(bestdistance_r, 1, return_indices=True)
                bestdistance_perclass = torch.flatten(bestdistance_perclass, start_dim=1)
                bestdistance_perclass = torch.flatten(F.adaptive_avg_pool1d(bestdistance_r, 1), start_dim=1)

                with torch.no_grad():
                    local_feat = torch.flatten(feat[0], start_dim=2)
                    for j, (lbl, idx1, idx2) in enumerate(zip(ys, indices.squeeze(), patch_idx.squeeze())):
                        local_counter[lbl, idx1[lbl]] += 1
                        eta = (1. / local_counter[lbl, idx1[lbl]]).to(device)
                        idx = model.npl * lbl + idx1[lbl]
                        local_clusters[idx,:,0,0] = (1-eta)*local_clusters[idx,:,0,0] + eta*local_feat[j:j+1,:,idx2[idx]]

                loss += F.cross_entropy(bestdistance_perclass, ys)
                loss += 1e-2 * F.pairwise_distance(model.prototype_layer_local.prototype_vectors.squeeze(), local_clusters.squeeze()).mean()
                ys_pred_local = torch.softmax(bestdistance_perclass.detach(), dim=1)

        ys_pred = ys_pred_global + ys_pred_local
        if torch.is_tensor(xg) and torch.is_tensor(xl):
            ys_pred = .5*ys_pred

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
        if torch.is_tensor(xg):
            total_acc_global += torch.sum(torch.eq(torch.argmax(ys_pred_global, dim=1), ys)).item() / float(len(xs))
        if torch.is_tensor(xl):
            total_acc_local += torch.sum(torch.eq(torch.argmax(ys_pred_local, dim=1), ys)).item() / float(len(xs))

        train_iter.set_postfix_str(
            f'Batch [{i + 1}/{len(train_loader)}], Loss: {total_loss/float(i+1):.3f}, Acc: {total_acc/float(i+1):.3f}'
        )

        if log is not None:
            log.log_values(log_loss, epoch, i + 1, loss.item(), acc)

    train_info['loss'] = total_loss / float(i + 1)
    train_info['train_accuracy'] = total_acc / float(i + 1)
    train_info['train_accuracy_global'] = total_acc_global / float(i + 1)
    train_info['train_accuracy_local'] = total_acc_local / float(i + 1)
    train_info['local_counter'] = local_counter
    train_info['global_counter'] = global_counter
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
    total_acc_global = 0.
    total_acc_local = 0.

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
        xg, xl = model(xs)

        loss = 0.
        ys_pred_global = 0.
        ys_pred_local = 0.
        if not model.global_prototypes and not model.local_prototypes:
            loss += F.cross_entropy(xg, ys)
            ys_pred_global = torch.softmax(xg.detach(), dim=1)
        else:
            if torch.is_tensor(xg):
                xg_r = xg.view(-1, model._num_classes, model.npl)
                # best_perclass = torch.flatten(F.adaptive_max_pool1d(xg_r, 1), start_dim=1)
                best_perclass = torch.flatten(F.adaptive_avg_pool1d(xg_r, 1), start_dim=1)
                loss += F.cross_entropy(best_perclass, ys)
                ys_pred_global = torch.softmax(best_perclass.detach(), dim=1)
            if torch.is_tensor(xl):
                xl_r = F.adaptive_max_pool2d(xl, (1,1)).squeeze().view(-1, model._num_classes, model.npl)
                # best_perclass = torch.flatten(F.adaptive_max_pool1d(xl_r, 1), start_dim=1)
                best_perclass = torch.flatten(F.adaptive_avg_pool1d(xl_r, 1), start_dim=1)
                loss += F.cross_entropy(best_perclass, ys)
                ys_pred_local = torch.softmax(best_perclass.detach(), dim=1)

        ys_pred = ys_pred_global + ys_pred_local
        if torch.is_tensor(xg) and torch.is_tensor(xl):
            ys_pred = .5*ys_pred
        ys_pred_max = torch.argmax(ys_pred, dim=1)

        # Update the confusion matrix
        for y_pred, y_true in zip(ys_pred_max, ys):
            cm[y_true][y_pred] += 1
        test_iter.set_postfix_str(
            f'Batch [{i + 1}/{len(test_iter)}], Acc: {acc_from_cm(cm):.3f}'
        )
        total_acc += torch.sum(torch.eq(torch.argmax(ys_pred, dim=1), ys)).item() / float(len(xs))
        if torch.is_tensor(xg):
            total_acc_global += torch.sum(torch.eq(torch.argmax(ys_pred_global, dim=1), ys)).item() / float(len(xs))
        if torch.is_tensor(xl):
            total_acc_local += torch.sum(torch.eq(torch.argmax(ys_pred_local, dim=1), ys)).item() / float(len(xs))
        probs.append(ys_pred.cpu())
        lbls.append(ys.cpu())

    info['confusion_matrix'] = cm
    info['test_bal_accuracy'] = acc_from_cm(cm)
    info['test_accuracy'] = total_acc / float(i + 1)
    info['test_accuracy_global'] = total_acc_global / float(i + 1)
    info['test_accuracy_local'] = total_acc_local / float(i + 1)
    if log is not None and not isinstance(log, str):
        log.log_message("\nEpoch %s - Test accuracy: " % (epoch) + str(info['test_bal_accuracy']))
        log.log_message("\nEpoch %s - Test confusion matrix: " % (epoch) + str(info['confusion_matrix']))
    elif isinstance(log, str):
        savepath = os.path.join(log, 'detections')
        if not os.path.isdir(savepath):
            os.makedirs(savepath)
        sio.savemat(os.path.join(savepath, 'maps.mat'), {'maps': torch.cat(maps, 0).numpy()})

    #######################
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
    #######################

    # reducer = umap.UMAP()
    # embedding = reducer.fit_transform(torch.cat(feats, 0))
    # plt.scatter(embedding[:, 0], embedding[:, 1])
    # plt.gca().set_aspect('equal', 'datalim')
    # plt.title('UMAP projection', fontsize=24)

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