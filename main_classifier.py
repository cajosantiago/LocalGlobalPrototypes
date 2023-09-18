from datetime import datetime
import os
from dataset_proto_dist import get_distloaders
import torch
import numpy as np

# from util.analyse import *
from log import Log

from train_classifier import eval_baseline, train_epoch_baseline
import my_models
from protonet_prototype_projection import project_with_class_constraints, save_projections
import scipy.io as sio

# os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
numworkers = 4

# Setup
dataset = '2019'#'CBIS_roi'#'CBIS'#
inputsize = 224 #if not dataset=='CBIS' else 512#512# 448#
n_epochs = 20
batch_size = 100
lr_classifier = 1e-2 #1e-1
npl = 10
npg = 10
optim_type = 'AdamW'#'SGD'#
n_classes = 8 if dataset=='2019' else 2
preload_data = True

device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
for net in ['resnet18','resnet50','vgg16','efficientnetb3','densenet169']:#['resnet18','resnet50','vgg16','efficientnetb3','densenet169']:#
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
            log_dir_pretrained = os.path.join('results', 'media_protonet_max_proj_kmeans_new', dataset, log_dir_suffix)
            log_dir = os.path.join('results', 'media_protonet_max_proj_kmeans_new_classifier', dataset, log_dir_suffix)
            print(log_dir)

            if (nfl==0 and nfg==0) :#or os.path.exists(log_dir + '/log.txt'):
                print('Already done!')
                continue

            ## Load pretrained model
            protonet = my_models.ProtoNet(net, npl, nfl, npg, nfg, n_classes)
            protonet.load_state_dict(torch.load(os.path.join(log_dir_pretrained, 'checkpoints', 'best_test_model', 'model_state.pt'), map_location='cpu'))
            protonet.checkpoint = log_dir_pretrained

            # Datasets
            train_loader, test_loader = get_distloaders(dataset, protonet, device, inputsize, batch_size, numworkers, preload_data)
            n_prototypes = train_loader.dataset.distances.size(1)
            n_classes = len(train_loader.dataset.classes)

            # Create a log for logging the loss values
            log = Log(log_dir)
            log_prefix = 'log_train_epochs'
            log_loss = log_prefix + '_losses'
            log.create_log(log_loss, 'epoch', 'batch', 'loss', 'batch_train_acc')
            log.create_log('log_epoch_overview', 'epoch', 'test_bal_acc', 'test_acc', 'train_acc', 'train_loss')
            log.create_log('log_best_test_cm', 'confusion matrix') ######

            # Classifier and optimizer
            model = my_models.Classifier(protonet, n_classes).to(device)
            optimizer = my_models.get_optimizer_classifier(model, optim_type, lr_classifier = lr_classifier)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[.5*n_epochs, .75*n_epochs], gamma=.1)

            best_train_acc = 0.
            best_test_acc = 0.
            best_test_auc = 0.
            ##################### Train classifier ########################
            for epoch in range(n_epochs):
                log.log_message("\nEpoch %s" % str(epoch))
                train_info = train_epoch_baseline(model, train_loader, optimizer, epoch, device, log)
                eval_info = eval_baseline(model, test_loader, epoch, device, log)
                scheduler.step()

                # Save stuff
                if eval_info['test_bal_accuracy'] > best_test_acc:
                    np.savetxt(os.path.join(log.log_dir, 'log_best_test_cm.csv'), eval_info['confusion_matrix'], delimiter=",", fmt='%d')
                    np.savetxt(os.path.join(log.log_dir, 'log_best_test_auc.csv'), np.array([eval_info['test_auc']]), delimiter=",", fmt='%f')
                    best_test_acc = eval_info['test_bal_accuracy']
                    directory_path = os.path.join(log.log_dir, 'checkpoints', 'best_test_model')
                    if not os.path.isdir(directory_path):
                        os.mkdir(directory_path)
                    torch.save(model.state_dict(), os.path.join(directory_path, 'model_state.pt'))
                if eval_info['test_auc'] > best_test_auc:
                    np.savetxt(os.path.join(log.log_dir, 'log_bestauc_test_cm.csv'), eval_info['confusion_matrix'], delimiter=",", fmt='%d')
                    np.savetxt(os.path.join(log.log_dir, 'log_bestauc_test_auc.csv'), np.array([eval_info['test_auc']]), delimiter=",", fmt='%f')
                    best_test_auc = eval_info['test_auc']
                    directory_path = os.path.join(log.log_dir, 'checkpoints', 'best_testauc_model')
                    if not os.path.isdir(directory_path):
                        os.mkdir(directory_path)
                    torch.save(model.state_dict(), os.path.join(directory_path, 'model_state.pt'))
                log.log_values('log_epoch_overview', epoch, eval_info['test_bal_accuracy'], eval_info['test_accuracy'], train_info['train_accuracy'], train_info['loss'])

print('Done!')
print(datetime.now())
