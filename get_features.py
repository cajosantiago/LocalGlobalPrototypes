from dataset import get_dataloaders
from tqdm import tqdm
import os
import torch
import my_models
import scipy.io as sio


@torch.no_grad()
def get_distances(model, device, loader):
    model.eval()
    model = model.to(device)

    # store distances
    features = torch.tensor(()).to(device)

    # Iterate through the test set
    print('Precomputing distances...')
    for i, (xs, ys) in tqdm(enumerate(loader), total=len(loader), ncols=0):
        xg, xl, feat = model(xs.to(device), get_features=True)
        features = torch.cat([features,feat[1]], 0)

    return features.cpu()

# os.environ["CUDA_VISIBLE_DEVICES"] = str(1)
numworkers = 4

# Setup
dataset = '2019'
inputsize = 224
batch_size = 100
npl = 10
npg = 10
n_classes = 8 if dataset=='2019' else 2
preload_data = True

# load image datasets
_, train_loader_noshuffle, _ = get_dataloaders(dataset, inputsize, batch_size, numworkers, preload_data)

device = torch.device('cuda:{}'.format(torch.cuda.current_device()))
for net in ['resnet18','resnet50','vgg16','efficientnetb3','densenet169']:#['resnet18','resnet50','vgg16','efficientnetb3','densenet169']:#
    for nfl in [0, 128, 256]:#[0, 128, 256]:#
        for nfg in [128, 256]:#[0, 128, 256]:#
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

            if (nfl==0 and nfg==0) :#or os.path.exists(log_dir + '/log.txt'):
                print('Already done!')
                continue

            ## Load pretrained model
            protonet = my_models.ProtoNet(net, npl, nfl, npg, nfg, n_classes)
            protonet.load_state_dict(torch.load(os.path.join(log_dir, 'checkpoints', 'best_test_model', 'model_state.pt'), map_location='cpu'))

            # run inference
            feat = get_distances(protonet, device, train_loader_noshuffle)

            # save features
            sio.savemat(os.path.join(log_dir,'global_features.mat'), {'f':feat.squeeze().numpy()})