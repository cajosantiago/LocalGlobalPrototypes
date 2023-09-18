from dataset import get_dataloaders
from tqdm import tqdm
import torch
import torch.nn.functional as F

def get_distloaders(dataset, model, device, inputsize=224, batch_size=64, numworkers=0, preload_data=True):

    # load image datasets
    _, train_loader_noshuffle, test_loader = get_dataloaders(dataset, inputsize, batch_size, numworkers, preload_data)

    # run inference
    train_dist = get_distances(model, device, train_loader_noshuffle)
    test_dist = get_distances(model, device, test_loader)

    # Datasets
    train_set = TensorDataset(train_dist, train_loader_noshuffle.dataset, 'train')
    test_set = TensorDataset(test_dist, test_loader.dataset, 'test')

    # Data loaders
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True,
                                               pin_memory=(torch.cuda.is_available()))#, num_workers=numworkers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False,
                                              pin_memory=(torch.cuda.is_available()))#, num_workers=numworkers)

    return train_loader, test_loader

class TensorDataset(torch.utils.data.Dataset):
    def __init__(self, distances, dataset, mode):
        self.distances = distances
        self.label = [item[1] for item in dataset.preloaded_data]
        self.classes = dataset.classes
        self.train = mode == 'train'

    def __getitem__(self, item):
        distance = self.distances[item]
        if self.train:
            distance += .01*torch.randn_like(distance)
        return distance, self.label[item]

    def __len__(self):
        return self.distances.size(0)

@torch.no_grad()
def get_distances(model, device, loader):
    model.eval()
    model = model.to(device)

    # store distances
    distances = torch.tensor(())
    total_acc = 0
    total_count = 0

    # Iterate through the test set
    print('Precomputing distances...')
    for i, (xs, ys) in tqdm(enumerate(loader), total=len(loader), ncols=0):
        xg, xl = model(xs.to(device))
        gd = xg.cpu() if torch.is_tensor(xg) else torch.tensor(())
        ld = F.adaptive_max_pool2d(xl, (1,1)).squeeze().cpu() if torch.is_tensor(xl) else torch.tensor(())
        d = torch.cat([gd, ld], 1)
        distances = torch.cat([distances, d], 0)

    #     # check performance
    #     d = torch.flatten(F.adaptive_avg_pool1d(d.view(-1, model._num_classes, 10), 1), start_dim=1)
    #     ys_pred_max = torch.argmax(d, dim=1)
    #     correct = torch.sum(torch.eq(ys_pred_max, ys)).item()
    #     total_acc += correct
    #     total_count += float(len(xs))
    # print(total_acc/total_count)

    return distances.cpu()
