import torch
from visdom import Visdom

from data import load_dataset
from model import Landmark
from resnet import resnext50_32x4d, resnet50
from util import print_basic_params
from configuration import CONF

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def finetune_resnet50():
    """3-stage finetune
        Step 1: freeze parameters of backbone to extract features
        Step 2: train last three dense `Linear-Linear-Softmax` layers
        Step 3: train bottom 9 layers, starting from `conv4_x`
    """

    model = resnet50(pretrained=True, num_classes=203094)
    modelname = 'resnet50_finetune'

    # Visualization
    vis = Visdom(env=modelname)

    # Landmark object
    landmark = Landmark(model, modelname, load_dataset, vis, device=device, epochs=15,
                        lr=1e-3,
                        optim_params={'name': 'sgd'})
    print_basic_params(landmark)

    try:
        landmark.resume(f"./checkpoints/{landmark.modelname}_best.ckpt")
    except FileNotFoundError:
        pass

    landmark.model = landmark.model.to(device)  # move the model parameters to CPU/GPU
    for e in range(landmark.cur_epoch, landmark.tot_epochs):
        landmark.cur_epoch = e + 1

        landmark.train()
        landmark.val()
        landmark.scheduler.step(landmark.best_acc)


def run():
    # Load model
    # model = resnext50_32x4d(num_classes=203094)
    # modelname = 'resnext50'

    model = resnet50(pretrained=False, num_classes=203094)
    modelname = 'resnet50'

    # Visualization
    vis = Visdom(env=modelname)

    # Landmark object
    landmark = Landmark(model, modelname, load_dataset, vis, device=device, epochs=CONF.tot_epochs,
                        lr=CONF.lr,
                        optim_params={'name': 'sgd'})
    print_basic_params(landmark)

    try:
        landmark.resume(f"./checkpoints/{landmark.modelname}_best.ckpt")
    except FileNotFoundError:
        pass

    landmark.model = landmark.model.to(device)  # move the model parameters to CPU/GPU
    for e in range(landmark.cur_epoch, landmark.tot_epochs):
        landmark.cur_epoch = e + 1
        for loader_index in range(len(landmark.loader_train_sets)):
            landmark.train(loader_index)
            landmark.val()
            landmark.scheduler.step(landmark.best_acc)


if __name__ == "__main__":
    run()
    # finetune_resnet50()
