import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from my_models import ProtoNet
import os
from skimage.util import view_as_blocks
from PIL import Image
import scipy
import json

@torch.no_grad()
def project_with_class_constraints(model: ProtoNet,
                                   loader,
                                   device,
                                   epoch,
                                   log,
                                   progress_prefix: str = 'Projection'
                                   ) -> dict:

    # Set the model to evaluation mode
    model.eval()
    torch.cuda.empty_cache()
    # The goal is to find the latent patch that minimizes the L2 distance to each prototype
    # To do this we iterate through the train dataset and store for each prototype the closest latent patch seen so far
    # Also store info about the image that was used for projection
    if model.local_prototypes:
        local_similarities = torch.tensor(())#[]
        local_features = torch.tensor(())#[]
    if model.global_prototypes:
        global_similarities = torch.tensor(())#[]
        global_features = torch.tensor(())#[]
    lbls = []

    # Build a progress bar for showing the status
    projection_iter = tqdm(enumerate(loader),
                           total=len(loader),
                           desc=progress_prefix,
                           ncols=0
                           )

    for i, (xs, ys) in projection_iter:
        lbls.extend(ys)

        xs, ys = xs.to(device), ys.to(device)
        # Perform a forward pass through the network
        XG, XL, FEAT = model(xs, get_features=True)

        if model.local_prototypes:
            local_similarities = torch.cat((local_similarities,XL.cpu()), 0)
            local_features = torch.cat((local_features,FEAT[0].cpu()), 0)

        if model.global_prototypes:
            global_similarities = torch.cat((global_similarities,XG.cpu()), 0)
            global_features = torch.cat((global_features,FEAT[1].cpu()), 0)

        projection_iter.set_postfix_str(f'Batch: {i + 1}/{len(loader)}')

    lbls = torch.stack(lbls, 0)
    if model.local_prototypes:
        for c in range(model._num_classes):
            ls_c = local_similarities[lbls==c][:, model.prototype_class_identity==c]
            feat_c = local_features[lbls==c]
            mls_c, patch_idx = F.adaptive_max_pool2d(ls_c, (1, 1), return_indices=True)
            col_ind, row_ind = scipy.optimize.linear_sum_assignment(torch.flatten(mls_c, start_dim=1).T.numpy(), maximize=True)
            prototype_update = torch.flatten(feat_c, start_dim=2)[row_ind,:,patch_idx[row_ind,col_ind].squeeze()]
            model.prototype_layer_local.prototype_vectors[model.prototype_class_identity==c,:,0,0] = prototype_update.to(device)

            # ls_c, _ = torch.flatten(local_similarities[lbls==c][:, model.prototype_class_identity==c], start_dim=2).max(dim=1)
            # feat_c = torch.flatten(local_features[lbls==c], start_dim=2)
            # _, patch_idx = ls_c.max(dim=1)
            # feat = feat_c[torch.arange(feat_c.size(0)), :, patch_idx]
            #
            # normal_feat = feat / torch.norm(feat, dim=1, keepdim=True).clamp_min(1e-12)
            # clusters = model.prototype_layer_local.prototype_vectors[model.prototype_class_identity == c].squeeze().cpu()
            # normal_clusters = clusters / torch.norm(clusters, dim=1, keepdim=True).clamp_min(1e-12)
            #
            # # project and update prototypes
            # sim = normal_clusters @ normal_feat.T
            # col_ind, row_ind = scipy.optimize.linear_sum_assignment(sim.T.numpy(), maximize=True)
            # new_clusters = normal_feat[row_ind,:]
            # model.prototype_layer_local.prototype_vectors[model.prototype_class_identity==c,:,0,0] = new_clusters.to(device)

        #save tsne
        model.plt_tsne_local.plot(torch.flatten(local_features.permute([0,2,3,1]), start_dim=0, end_dim=2),
                      torch.flatten(lbls.unsqueeze(1).unsqueeze(2).expand(-1,local_features.shape[2],local_features.shape[3])),
                      model.prototype_layer_local.prototype_vectors.squeeze().cpu(), model.prototype_class_identity)
        model.plt_tsne_local.save_pdf(os.path.join(log.log_dir, 'tsne', 'local'), epoch)


    if model.global_prototypes:
        for c in range(model._num_classes):
            gs_c = global_similarities[lbls==c][:,model.global_prototype_class_identity==c]
            feat_c = global_features[lbls==c]
            row_ind, col_ind = scipy.optimize.linear_sum_assignment(torch.flatten(gs_c, start_dim=1).T.numpy(), maximize=True)
            model.prototype_layer_global.prototype_vectors[model.global_prototype_class_identity==c] = feat_c[row_ind].to(device)

            # feat = global_features[lbls==c].squeeze()
            # normal_feat = feat / torch.norm(feat, dim=1, keepdim=True).clamp_min(1e-12)
            # clusters = model.prototype_layer_global.prototype_vectors[model.global_prototype_class_identity == c].squeeze().cpu()
            # normal_clusters = clusters / torch.norm(clusters, dim=1, keepdim=True).clamp_min(1e-12)
            #
            # # project and update prototypes
            # sim = normal_clusters @ normal_feat.T
            # col_ind, row_ind = scipy.optimize.linear_sum_assignment(sim, maximize=True)
            # new_clusters = normal_feat[row_ind,:]
            # model.prototype_layer_global.prototype_vectors[model.global_prototype_class_identity==c,:,0,0] = new_clusters.to(device)

        # save tsne
        model.plt_tsne_global.plot(global_features.squeeze(), lbls,
                                   model.prototype_layer_global.prototype_vectors.squeeze().cpu(),
                                   model.global_prototype_class_identity)
        model.plt_tsne_global.save_pdf(os.path.join(log.log_dir, 'tsne', 'global'), epoch)

@torch.no_grad()
def save_projections(model: ProtoNet,
                       loader,
                       device,
                       log_dir,
                       progress_prefix: str = 'Projection'
                       ) -> dict:

    dataset = loader.dataset
    # Set the model to evaluation mode
    model.eval()
    torch.cuda.empty_cache()
    # The goal is to find the latent patch that minimizes the L2 distance to each prototype
    # To do this we iterate through the train dataset and store for each prototype the closest latent patch seen so far
    # Also store info about the image that was used for projection
    if model.local_prototypes:
        local_features = []
        num_local_prototypes = model.prototype_layer_local.prototype_vectors.shape[0]
        best_local_sim = [-1e6] * num_local_prototypes
        local_counter = torch.zeros(num_local_prototypes, model._num_classes)
        local_min_info = {j: None for j in range(num_local_prototypes)}
        path_local = os.path.join(log_dir, 'projections', 'local_prototypes')
        if not os.path.isdir(path_local):
            os.makedirs(path_local)
    if model.global_prototypes:
        global_features = []
        num_global_prototypes = model.prototype_layer_global.prototype_vectors.shape[0]
        best_global_sim = [-1e6] * num_global_prototypes
        global_counter = torch.zeros(num_global_prototypes, model._num_classes)
        global_min_info = {j: None for j in range(num_global_prototypes)}
        path_global = os.path.join(log_dir, 'projections', 'global_prototypes')
        if not os.path.isdir(path_global):
            os.makedirs(path_global)

    # Build a progress bar for showing the status
    projection_iter = tqdm(enumerate(loader),
                           total=len(loader),
                           desc=progress_prefix,
                           ncols=0
                           )

    for i, (xs, ys) in projection_iter:
        # Get a batch of data
        xs, ys = xs.to(device), ys.to(device)

        # Perform a forward pass through the network
        XG, XL, FEAT = model(xs, get_features=True)

        if model.local_prototypes:
            local_features.append(FEAT[0].cpu())
            bestdistance = F.adaptive_max_pool2d(XL, (1,1)).view(-1, model._num_classes, model.npl)
            _, idxs = F.adaptive_max_pool1d(bestdistance, 1, return_indices=True)
            idxs = torch.flatten(idxs, start_dim=1) + (torch.arange(model._num_classes)*model.npl).to(device)
            for j in range(idxs.shape[0]):
                for k in range(idxs.shape[1]):
                    local_counter[idxs[j,k], ys[j]] += 1

            BS, NLP, W, H = XL.shape
            for j in range(num_local_prototypes):
                xl = torch.flatten(XL[:,j], start_dim=1)
                proto_class = model.prototype_class_identity[j]
                xl_aux = xl[ys==proto_class]
                indices = (ys==proto_class).nonzero().squeeze()
                if xl_aux.shape[0] > 0:
                    local_sim = xl_aux.max()
                    idx = torch.nonzero(xl_aux == local_sim)[0]
                    idx[0] = indices[idx[0]]
                    if local_sim > best_local_sim[j]:
                        best_local_sim[j] = local_sim
                        img_idx = (i * BS + idx[0]).item()
                        local_min_info[j] = {
                            'input_image_ix': img_idx,
                            'patch_ix': idx[1].item(),
                            'W': W,
                            'H': H,
                            'W1': 1,
                            'H1': 1,
                            'distance': local_sim.item(),
                            'nearest_image': dataset.samples[img_idx][0],
                            'class': dataset.classes[proto_class.int()],
                            'class_confirmation': dataset.samples[img_idx][1]
                        }


        if model.global_prototypes:
            global_features.append(FEAT[1].cpu())
            _, idxs = F.adaptive_max_pool1d(XG.view(-1, model._num_classes, model.npg), 1, return_indices=True)
            idxs = torch.flatten(idxs, start_dim=1) + (torch.arange(model._num_classes)*model.npg).to(device)
            for j in range(idxs.shape[0]):
                for k in range(idxs.shape[1]):
                    global_counter[idxs[j,k], ys[j]] += 1

            BS = XG.shape[0]
            for j in range(num_global_prototypes):
                xg = XG[:,j]
                proto_class = model.global_prototype_class_identity[j]
                xg_aux = xg[ys==proto_class]
                indices = (ys==proto_class).nonzero().squeeze()
                if xg_aux.shape[0] > 0:
                    global_sim = xg_aux.max()
                    idx = indices[xg_aux.argmax()]
                    if global_sim > best_global_sim[j]:
                        best_global_sim[j] = global_sim
                        img_idx = (i * BS + idx).item()
                        global_min_info[j] = {
                            'input_image_ix': img_idx,
                            'distance': global_sim.item(),
                            'nearest_image': dataset.samples[img_idx][0],
                            'class': dataset.classes[proto_class.int()],
                            'class_confirmation': dataset.samples[img_idx][1]
                        }
        projection_iter.set_postfix_str(f'Batch: {i + 1}/{len(loader)}')

    if model.local_prototypes:
        np.savetxt(os.path.join(path_local, 'counter.csv'), local_counter.numpy(), delimiter=",", fmt='%d')
        with open(os.path.join(path_local, 'local_projection.json'), 'w') as f:
            f.write(json.dumps(local_min_info))
        for j in local_min_info:
            dict = local_min_info[j]
            path_local_j = os.path.join(path_local, dict['class'])
            if not os.path.isdir(path_local_j):
                os.makedirs(path_local_j)
            img = dataset.loader(dict['nearest_image'])
            img.save(os.path.join(path_local_j, 'local_prototype_%s_%d_fullimage.jpg' %
                                  (dict['class'], j%model.npl)))
            img = img.resize((img.width-img.width%dict['W'],img.height-img.height%dict['H']))
            patch_size = (int(img.height / dict['H']), int(img.width / dict['W']), len(img.getbands())) if len(
                img.getbands()) == 3 else (int(img.height / dict['H']), int(img.width / dict['W']))
            img_patches = view_as_blocks(np.array(img), patch_size)
            patch = Image.fromarray(img_patches[dict['patch_ix'] // dict['W'], dict['patch_ix'] % dict[
                'W'], 0]) if len(img.getbands()) == 3 else Image.fromarray(
                img_patches[dict['patch_ix'] // dict['W'], dict['patch_ix'] % dict['W']])
            patch.save(os.path.join(path_local_j, 'local_prototype_%s_%d_patch.jpg' % (dict['class'], j%model.npl)))

    if model.global_prototypes:
        np.savetxt(os.path.join(path_global, 'counter.csv'), global_counter.numpy(), delimiter=",", fmt='%d')
        with open(os.path.join(path_global, 'global_projection.json'), 'w') as f:
            f.write(json.dumps(global_min_info))
        for j in global_min_info:
            dict = global_min_info[j]
            path_global_j = os.path.join(path_global, dict['class'])
            if not os.path.isdir(path_global_j):
                os.makedirs(path_global_j)
            img = dataset.loader(dict['nearest_image'])
            img.save(os.path.join(path_global_j, 'global_prototype_%s_%d.jpg' %
                                  (dict['class'], j%model.npg)))
