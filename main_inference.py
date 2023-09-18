from datetime import datetime
import os
from dataset import get_singleimage
import torch
import my_models
import sys
import scipy.io as sio

os.environ["CUDA_VISIBLE_DEVICES"] = ''
device = torch.device('cuda:{}'.format(torch.cuda.current_device())) if torch.cuda.is_available() else 'cpu'

# Inputs
modelfolder = sys.argv[1]
imagefile = sys.argv[2]
outputfolder = sys.argv[3]

# Get info
npl = 10
npg = 10
parts = os.path.normpath(modelfolder).split(os.path.sep)[-1].split('_')
net = parts[0]
if parts[1] == 'joint':
    nfl = int(parts[2][1:])
    nfg = int(parts[3][1:])
    n_prototypes = npl + npg
elif parts[1] == 'global':
    nfg = int(parts[2][1:])
    nfl = 0
    n_prototypes = npg
else:
    nfl = int(parts[2][1:])
    nfg = 0
    n_prototypes = npl
n_classes = 8

# Model
protopnet = my_models.ProtoNet(net, npl, nfl, npg, nfg, n_classes).to(device)
protopnet.load_state_dict(torch.load(os.path.join(modelfolder,'checkpoints/best_test_model/model_state.pt'),map_location=torch.device('cpu')))
protopnet.checkpoint = modelfolder
classifier = my_models.Classifier(protopnet, n_classes).to(device)
mfparts = os.path.normpath(modelfolder).split(os.path.sep)
mfparts[3] = 'media_protonet_max_proj_kmeans_new_classifier'
state_dict = torch.load(os.path.join(os.path.sep.join(mfparts),'checkpoints/best_test_model/model_state.pt'),map_location=torch.device('cpu'))
classifier.load_state_dict(state_dict)

# Run inference
protopnet.eval()
classifier.eval()
with torch.no_grad():
    xg, xl, feat = protopnet(get_singleimage(imagefile).to(device), get_features=True)
    gd = xg if torch.is_tensor(xg) else torch.tensor(()).to(device)
    ld = torch.flatten(torch.nn.functional.adaptive_max_pool2d(xl, (1, 1)), start_dim=1) if torch.is_tensor(xl) else torch.tensor(()).to(device)
    d = torch.cat([gd, ld], 1)
    probs = torch.softmax(classifier(d), dim=1)

sio.savemat(os.path.join(outputfolder, 'output.mat'),
            {'local_dist':xl.cpu().numpy(), #nao da se vier a vazio!
             'global_dist':gd.cpu().numpy(),
             'probs':probs.cpu().numpy(),
             'classifier_weights':(state_dict['W']*state_dict['mask']).squeeze().cpu().numpy(),
             'feat': feat[1].cpu().numpy()})
