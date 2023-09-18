from datetime import datetime
import os
from dataset import get_dataloaders
import torch
import numpy as np

# from util.analyse import *
from log import Log

from train_protonet import eval_baseline, train_epoch_baseline
import my_models
from protonet_prototype_projection import project_with_class_constraints, save_projections

# os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
numworkers = 4

# Setup
dataset = '2019'#'CBIS_roi'#'CBIS'#
inputsize = 224 #if not dataset=='CBIS' else 512#512# 448#
net = 'vgg16'#'densenet169'#'efficientnetb3'#'resnet18'#'resnet50'#
n_epochs = 50 if 'CBIS' in dataset else 100
batch_size = 50#60#50 if not dataset=='CBIS' else 12
lr_net = 1e-5
lr_block_local = 1e-3
lr_block_global = 1e-3
lr_local = 1e-2
lr_global = 1e-2
npl = 10
npg = 10
optim_type = 'AdamW'#'SGD'#
project_prototypes_every = 10 #epochs
preload_data = False
# if preload_data:
#     numworkers = 0

# Datasets
train_loader, train_loader_noshuffle, test_loader = get_dataloaders(dataset, inputsize, batch_size, numworkers, preload_data)
n_classes = len(train_loader.dataset.classes)

for net in ['efficientnetb3','densenet169']:#['resnet18','resnet50','vgg16','efficientnetb3','densenet169']:#
    for nfl in [0, 128, 256]:#[0, 128, 256]:#
        for nfg in [0, 128, 256]:#[0, 128, 256]:#
                if nfl == 0 and nfg == 0:
                    log_dir_suffix = '%s_baseline' % net
                elif nfl == 0:
                    log_dir_suffix ='%s_global_g%d' % (net, nfg)
                elif nfg == 0:
                    log_dir_suffix = '%s_local_l%d' % (net, nfl)
                else:
                    log_dir_suffix = '%s_joint_l%d_g%d' % (net, nfl, nfg)
                log_dir = os.path.join('results', 'media_protonet_max_proj_kmeans_new', dataset, log_dir_suffix)
                print(log_dir)

                if os.path.exists(log_dir + '/log.txt') or (nfl==0 and nfg==0):
                    print('Already done!')
                    continue

                ## Model
                model = my_models.ProtoNet(net, npl, nfl, npg, nfg, n_classes)
                device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
                model = model.to(device)

                # # resume training
                # model.load_state_dict(torch.load(os.path.join(log_dir, 'checkpoints', 'best_test_model', 'model_state.pt')))

                # Create a log for logging the loss values
                log = Log(log_dir)
                log_prefix = 'log_train_epochs'
                log_loss = log_prefix + '_losses'
                log.create_log(log_loss, 'epoch', 'batch', 'loss', 'batch_train_acc')
                log.create_log('log_epoch_overview', 'epoch', 'test_bal_acc', 'test_acc', 'test_acc_global', 'test_acc_local',
                               'train_acc', 'train_acc_global', 'train_acc_local', 'train_loss')
                log.create_log('log_best_test_cm', 'confusion matrix') ######

                # optimizer
                optimizer = my_models.get_optimizer_protonet(model, optim_type,
                                                           lr_net = lr_net, lr_block_local = lr_block_local,
                                                           lr_block_global = lr_block_global, lr_local = lr_local,
                                                           lr_global = lr_global)
                # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[.7*n_epochs, .9*n_epochs], gamma=.1)

                best_train_acc = 0.
                best_test_acc = 0.
                best_test_auc = 0.

                ##################### Train the model ########################
                for epoch in range(n_epochs):

                    if epoch % project_prototypes_every == 0:
                        project_with_class_constraints(model, train_loader_noshuffle, device, epoch, log)

                    log.log_message("\nEpoch %s" % str(epoch))

                    train_info = train_epoch_baseline(model, train_loader, optimizer, epoch, device, log)

                    eval_info = eval_baseline(model, test_loader, epoch, device, log)

                    best_test_acc, best_test_auc = log.save_stuff(epoch, train_info, eval_info, best_test_acc, best_test_auc, model)

                    # scheduler.step()

                log.log_message("Training Finished. Best training accuracy was %s, best test accuracy was %s\n" % (
                str(best_train_acc), str(best_test_acc)))

                if model.global_prototypes or model.local_prototypes:
                    # load best model
                    model.load_state_dict(torch.load(os.path.join(log_dir, 'checkpoints', 'best_test_model', 'model_state.pt')))
                    # Get prototype projections
                    save_projections(model, train_loader_noshuffle, device, log_dir)

print('Done!')
print(datetime.now())
