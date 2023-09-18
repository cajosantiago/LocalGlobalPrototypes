import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import torch
import os


class show_features():
    def __init__(self):
        self.init = []
        self.seed = 0

    def plot(self, feat, lbls, prototypes, protolbls):

        # select subset of samples
        N = 2000
        if feat.shape[0] > N:
            rndperm = np.random.RandomState(seed=self.seed).permutation(feat.shape[0])[1:N]
            feat = feat[rndperm]
            lbls = lbls[rndperm]

        n_prototypes = prototypes.shape[0]
        n_classes = int(protolbls.max() + 1)

        X = torch.cat((feat,prototypes), 0)
        init = self.init if len(self.init) > 0 else 'random'
        tsne = TSNE(n_components=2, metric='cosine', n_jobs=-1, square_distances=True, init=init)
        feat2d = tsne.fit_transform(X.detach().numpy())
        self.init = feat2d

        # separate variables
        proto = feat2d[-n_prototypes:]
        labels_proto = protolbls.numpy()
        feat2d = feat2d[:-n_prototypes]
        labels = lbls.numpy()

        plt.figure(1)
        plt.clf()
        for label_id in range(n_classes):
            plt.scatter(feat2d[np.where(labels == label_id), 0],
                        feat2d[np.where(labels == label_id), 1],
                        marker='o',
                        color=plt.cm.Set1(label_id / float(n_classes)),
                        linewidth=1,
                        alpha=0.1,
                        label=label_id)
            plt.scatter(proto[np.where(labels_proto == label_id), 0],
                        proto[np.where(labels_proto == label_id), 1],
                        marker='X',
                        color=plt.cm.Set1(label_id / float(n_classes)),
                        linewidth=1)
        plt.legend(loc='best')
        manager = plt.get_current_fig_manager()
        manager.window.maximize()
        # plt.pause(.1)
        #plt.show()

    def save_pdf(self, folder, epoch):
        os.makedirs(folder, exist_ok=True)
        plt.savefig(os.path.join(folder, 'tsne_%03d.pdf' %epoch))
        plt.close()