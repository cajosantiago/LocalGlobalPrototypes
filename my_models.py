import torch
from torch import nn
from torchvision import models
import torch.nn.functional as F
import os
import pickle
import json

# from util.l2conv import L2Conv2D
from visualization import show_features

def backbone_model(backbone):
    if backbone == 'vgg16':
        model = models.vgg16(pretrained=True)
        features = nn.Sequential(*list(model.features.children())[:-1])
        in_channels = 512
    if backbone == 'resnet18':
        model = models.resnet18(pretrained=True)
        features = nn.Sequential(*list(model.children())[:-2])
        in_channels = 512
    if backbone == 'resnet50':
        model = models.resnet50(pretrained=True)
        features = nn.Sequential(*list(model.children())[:-2])
        in_channels = 2048
    if backbone == 'densenet169':
        model = models.densenet169(pretrained=True)
        features = nn.Sequential(*list(model.children())[:-1])
        in_channels = 1664
    if backbone == 'efficientnetb3':
        model = models.efficientnet_b3(pretrained=True)
        features = model.features
        in_channels = 1536

    return features, in_channels

class Prototype_Layer(nn.Module):
    def __init__(self, num_prototypes, num_features, w_1, h_1):
        super().__init__()
        prototype_shape = (num_prototypes, num_features, w_1, h_1)
        # self.prototype_vectors = nn.Parameter(torch.randn(prototype_shape), requires_grad=True)
        self.prototype_vectors = nn.Parameter(torch.eye(num_prototypes,num_features).view(prototype_shape), requires_grad=True)
        # self.prototype_vectors = nn.Parameter(torch.eye(num_prototypes,num_features).view(prototype_shape), requires_grad=False)
        self.eps = 1e-12
    def forward(self, x):
        xn = x / torch.norm(x, dim=1, keepdim=True).clamp_min(self.eps)
        pn = self.prototype_vectors / torch.norm(self.prototype_vectors, dim=1, keepdim=True).clamp_min(self.eps)
        return (xn[:,None,:,:,:] * pn[None,:,:,:,:]).sum(dim=2)

class ProtoNet(nn.Module):
    def __init__(self, backbone, npl, nfl, npg, nfg, n_classes, dropout=0):
        super().__init__()
        self.local_prototypes = nfl > 0
        self.global_prototypes = nfg > 0
        self.nfl = nfl
        self.nfg = nfg
        self.npl = npl
        self.npg = npg
        self._num_classes = n_classes
        self.dropout = dropout
        self.temp = 0.07

        features, in_channels = backbone_model(backbone)
        self._net = features

        if self.local_prototypes:
            self.plt_tsne_local = show_features()
            self.prototype_class_identity = torch.zeros(npl*n_classes)
            for j in range(npl*n_classes):
                self.prototype_class_identity[j] = j // npl
            self._add_on_local = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=nfl, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=nfl, out_channels=nfl, kernel_size=1),
                nn.Sigmoid()
            )
            self.prototype_layer_local = Prototype_Layer(npl*n_classes, nfl, 1, 1)
        if self.global_prototypes:
            self.plt_tsne_global = show_features()
            self.global_prototype_class_identity = torch.zeros(npg*n_classes)
            for j in range(npg*n_classes):
                self.global_prototype_class_identity[j] = j // npg
            self._add_on_global = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Conv2d(in_channels=in_channels, out_channels=nfg, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(in_channels=nfg, out_channels=nfg, kernel_size=1),
                nn.Sigmoid()
            )
            self.prototype_layer_global = Prototype_Layer(npg*n_classes, nfg, 1, 1)
        if not self.global_prototypes and not self.local_prototypes:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(),
                nn.Linear(in_channels, n_classes),
            )

    def forward(self, x, get_features=False):
        feat = self._net(x)

        xl = []
        local_feat = []
        xg = []
        global_feat = []
        if self.local_prototypes:
            local_feat = self._add_on_local(feat)
            xl = self.prototype_layer_local(local_feat)
            if self.training:
                xl = xl/self.temp
                if self.dropout > 0:
                    prob = torch.rand(xl.shape[1])
                    for c in range(self._num_classes): # must have at least one prototype per class
                        idx = prob[self.prototype_class_identity==c].argmax()
                        prob[self.prototype_class_identity==c][idx] = 2 #always >self.dropout
                    xl[:,prob<self.dropout,:,:] = -1/self.temp

        if self.global_prototypes:
            global_feat = self._add_on_global(feat)
            xg = torch.flatten(self.prototype_layer_global(global_feat), start_dim=1)
            if self.training:
                xg = xg/self.temp
                if self.dropout > 0:
                    prob = torch.rand(xg.shape[1])
                    for c in range(self._num_classes): # must have at least one prototype per class
                        idx = prob[self.global_prototype_class_identity==c].argmax()
                        prob[self.global_prototype_class_identity==c][idx] = 2 #always >self.dropout
                    xg[:,prob<self.dropout] = -1/self.temp
        if not self.global_prototypes and not self.local_prototypes:
            xg = self.classifier(feat)

        if get_features:
            return xg, xl, (local_feat, global_feat)
        else:
            return xg, xl

    # def update_prototypes(self, feat, sim, mode):
    #     if mode == 'local':
    #         sim_p, closest_prototype = F.adaptive_max_pool1d(sim.permute((0,2,3,1)), 3, return_indices=True)
    #         _, closest_patch = F.adaptive_max_pool1d(torch.flatten(sim_p,start_dim=1), 1, return_indices=True)
    #         feat_p = torch.flatten(feat, start_dim=2)[:,:,closest_patch]
    #         new_proto = torch.zeros_like(self.prototype_layer_local.prototype_vectors)
    #         for p in range(self.npl * self._num_classes):
    #             new_proto[p] = torch.mean(feat_p[closest_prototype.squeeze() == p], dim=0)


def get_optimizer_protonet(model, optim_type, lr_net=1e-5, lr_block_local=1e-5, lr_block_global=1e-5, lr_local=1e-5, lr_global=1e-5, weight_decay=0, momentum=0.9):
    paramlist = [{"params": model._net.parameters(), "lr": lr_net, "weight_decay_rate": weight_decay}]
    if model.local_prototypes:
        paramlist.append({"params": model._add_on_local.parameters(), "lr": lr_block_local, "weight_decay_rate": weight_decay,"momentum": momentum})
        paramlist.append({"params": model.prototype_layer_local.parameters(), "lr": lr_local, "weight_decay_rate": 0})
    if model.global_prototypes:
        paramlist.append({"params": model._add_on_global.parameters(), "lr": lr_block_global, "weight_decay_rate": weight_decay,"momentum": momentum})
        paramlist.append({"params": model.prototype_layer_global.parameters(), "lr": lr_global, "weight_decay_rate": 0})
    if not model.global_prototypes and not model.local_prototypes:
        paramlist.append({"params": model.classifier.parameters(), "lr": lr_block_global, "weight_decay_rate": weight_decay,"momentum": momentum})

    if optim_type == 'SGD':
        return torch.optim.SGD(paramlist)
    if optim_type == 'Adam':
        return torch.optim.Adam(paramlist)
    if optim_type == 'AdamW':
        return torch.optim.AdamW(paramlist)

class Classifier(nn.Module):
    def __init__(self, protonet, n_classes):
        super().__init__()

        # Create mask to remove prototypes that are projected to the same image/patch
        ignore_local, ignore_global = get_repeated_prototypes(protonet)
        local_class_weight = [protonet.npl-sum([1 for i in ignore_local if i // protonet.npl == c]) for c in range(n_classes)]
        global_class_weight = [protonet.npg-sum([1 for i in ignore_global if i // protonet.npg == c]) for c in range(n_classes)]

        # Initialize weight matrix
        init_local, init_global = torch.tensor([]), torch.tensor([])
        mask_local, mask_global = torch.tensor([]), torch.tensor([])
        if protonet.local_prototypes:
            init_local = torch.zeros((n_classes, protonet.npl * n_classes))
            mask_local = torch.ones((n_classes, protonet.npl * n_classes))
            for i in range(init_local.shape[0]):
                for j in range(init_local.shape[1]):
                    if (j // protonet.npl) == i:
                        init_local[i, j] = 1./local_class_weight[i]
                    if i == 0 and j in ignore_local:
                            mask_local[:,j] = 0
        if protonet.global_prototypes:
            init_global = torch.zeros((n_classes, protonet.npg * n_classes))
            mask_global = torch.ones((n_classes, protonet.npl * n_classes))
            for i in range(init_global.shape[0]):
                for j in range(init_global.shape[1]):
                    if (j // protonet.npg) == i:
                        init_global[i, j] = 1./global_class_weight[i]
                    if i == 0 and j in ignore_global:
                            mask_global[:,j] = 0

        self.W = torch.nn.Parameter(torch.cat((init_local, init_global), 1))
        self.mask = torch.nn.Parameter(torch.cat((mask_local, mask_global), 1))

        # n_prototypes = [protonet.npl if protonet.local_prototypes else 0,
        #                 protonet.npg if protonet.global_prototypes else 0]
        # init = torch.zeros_like(self.classifier.weight)
        # for i, v in enumerate(init):
        #     for j, _ in enumerate(v):
        #         if (j % (n_classes * 80)) // 10 == i:
        #             init[i, j] = 1/10
        # self.classifier = nn.Linear(n_classes * (n_prototypes[0]+n_prototypes[1]), n_classes, bias=False)
        # self.classifier.weight = torch.nn.Parameter(init)
        self._num_classes = n_classes


    def forward(self, x):
        # return self.classifier(x)
        return x @ (self.W * self.mask).T

def get_optimizer_classifier(model, optim_type, lr_classifier=1e-5, weight_decay=0, momentum=0.9):
    # paramlist = [{"params": model.parameters(), "lr": lr_classifier, "weight_decay_rate": weight_decay}]
    paramlist = [{"params": [model.W], "lr": lr_classifier, "weight_decay_rate": weight_decay}]
    if optim_type == 'SGD':
        return torch.optim.SGD(paramlist)
    if optim_type == 'Adam':
        return torch.optim.Adam(paramlist)
    if optim_type == 'AdamW':
        return torch.optim.AdamW(paramlist)


def get_repeated_prototypes(protonet):
    ignore_global, ignore_local = [], []
    checkpointdir = protonet.checkpoint
    if protonet.global_prototypes and os.path.exists(os.path.join(checkpointdir, 'projections', 'global_prototypes', 'global_projection.json')):
        with open(os.path.join(checkpointdir, 'projections', 'global_prototypes', 'global_projection.json')) as f:
            js = json.load(f)
        for i in range(len(js)):
            if not i in ignore_global:
                proj_image = js[str(i)]['nearest_image']
                simi = js[str(i)]['distance']
                for j in range(i+1, len(js)):
                    if proj_image == js[str(j)]['nearest_image']:
                        if simi <= js[str(j)]['distance']:
                            ignore_global.append(i)
                            break
                        else:
                            ignore_global.append(j)

    if protonet.local_prototypes and os.path.exists(os.path.join(checkpointdir, 'projections', 'local_prototypes', 'local_projection.json')):
        with open(os.path.join(checkpointdir, 'projections', 'local_prototypes', 'local_projection.json')) as f:
            js = json.load(f)
        for i in range(len(js)):
            if not i in ignore_local:
                proj_image = js[str(i)]['nearest_image']
                patch_idx = js[str(i)]['patch_ix']
                simi = js[str(i)]['distance']
                for j in range(i+1, len(js)):
                    if proj_image == js[str(j)]['nearest_image'] and patch_idx == js[str(j)]['patch_ix']:
                        if simi <= js[str(j)]['distance']:
                            ignore_local.append(i)
                            break
                        else:
                            ignore_local.append(j)

    return ignore_local, ignore_global