from datetime import datetime
import os
from dataset import get_testloader
import torch

from util.analyse import *
# from util.save import *

from train_protonet import eval_baseline, train_epoch_baseline
import my_models
from protonet_prototype_projection import project_with_class_constraints, save_projections

# os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
numworkers = 0

# Setup
dataset = '2019'#'CBIS_roi'#'CBIS'#
inputsize = 224 if not dataset=='CBIS' else 512#512# 448#
batch_size = 100
npl = 10

# Datasets
test_loader = get_testloader(dataset, inputsize, batch_size, numworkers)
n_classes = 8 if dataset=='2019' else 2 #len(test_loader.dataset.classes)

for net in ['resnet18']:#['resnet18','resnet50','vgg16','densenet169','efficientnetb3']:
    for nfl in [128]:#[0, 128, 256]:
        for nfg in [0]:#[0, 128, 256]:
                if nfl == 0 and nfg == 0:
                    log_dir_suffix = '%s_baseline' % net
                elif nfl == 0:
                    log_dir_suffix ='%s_global_g%d' % (net, nfg)
                elif nfg == 0:
                    log_dir_suffix = '%s_local_l%d' % (net, nfl)
                else:
                    log_dir_suffix = '%s_joint_l%d_g%d' % (net, nfl, nfg)
                log_dir = os.path.join('results', 'media_protonet_max_proj', dataset, log_dir_suffix)
                print(log_dir)


                ## Model
                model = my_models.ProtoNet(net, npl, nfl, nfg, n_classes)
                if 'CBIS' in dataset:
                    model.load_state_dict(torch.load(os.path.join(log_dir,'checkpoints/best_testauc_model/model_state.pt')))
                else:
                    model.load_state_dict(torch.load(os.path.join(log_dir,'checkpoints/best_test_model/model_state.pt')))
                device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
                model = model.to(device)

                eval_info = eval_baseline(model, test_loader, 'final', device, log=log_dir)

                ################### save stuff
                np.savetxt(os.path.join(log_dir,'final_test_cm.csv'), eval_info['confusion_matrix'], delimiter=",", fmt='%d')
                np.savetxt(os.path.join(log_dir,'final_test_auc.csv'), np.array([eval_info['test_auc']]), delimiter=",", fmt='%f')
                np.savetxt(os.path.join(log_dir,'final_test_probs.csv'), eval_info['probs'], delimiter=",", fmt='%f')
                ###################

print('Done!')
print(datetime.now())
