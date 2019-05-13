import torch
from visdom import Visdom
import torch.hub

from data import load_dataset
from model import Landmark
from resnet import resnext50_32x4d
from util import print_basic_params

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

NUM_CLASSES = 203094

def run():
    # Load model
    # model = resnext50_32x4d(num_classes=203094)
    # modelname = 'resnext50'
    model = torch.hub.load(
            'moskomule/senet.pytorch',
            'se_resnet20',
            num_classes=NUM_CLASSES)
    modelname = 'se_resnet20'


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


if __name__ == "__main__":
    run()
