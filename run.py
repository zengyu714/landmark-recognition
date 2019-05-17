"""
Based on ResNet50
"""
import argparse
import json

import torch
from visdom import Visdom

from data import load_dataset
from landmark import Landmark
from utils.util import print_basic_params, unfreeze_resnet50_bottom

parser = argparse.ArgumentParser(description='Google Landmark Recognition Challenge')
parser.add_argument('-g', '--cuda-device', type=int, default=0,
                    help='Choose which gpu to use (default: 0)')
parser.add_argument('-f', '--finetune', action="store_true",
                    help='Finetune the resnet50')
parser.add_argument('--model-name', type=str, default="finetune",
                    help='The name of the finetune model')
parser.add_argument('--use-stage', type=int, default=0,
                    help='Use 3 stage strategy to finetune')
parser.add_argument('--stage-epoch', type=int, default=3,
                    help='The epoch to finetune the fc layers')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate')
parser.add_argument('--optim-params', type=str, default='{"name": "adam"}',
                    help='The name of optimizer, default is adam')
parser.add_argument('--tot-epochs', type=int, default=9,
                    help='Total training epochs')
parser.add_argument('--batch-size', type=int, default=512,
                    help='Batch size')

args = parser.parse_args()
args.optim_params = json.loads(args.optim_params)

if torch.cuda.is_available():
    torch.cuda.set_device(args.cuda_device)
    device = torch.cuda.current_device()
else:
    device = torch.device('cpu')


def finetune_stage():
    """3-stage finetune
        Step 1: freeze parameters of backbone to extract features
        Step 2: train last three dense `Linear-Linear-Softmax` layers
        Step 3: train bottom 9 layers, starting from `conv4_x`
    """

    modelname = args.model_name

    # Visualization
    vis = Visdom(env=modelname)

    # Landmark object
    landmark = Landmark(modelname, load_dataset, vis,
                        pretrained=True,  # use pretrained model
                        use_stage=True,
                        device=device,
                        lr=args.lr,
                        epochs=args.tot_epochs,
                        batch_size=args.batch_size,
                        optim_params=args.optim_params)

    print_basic_params(landmark)

    try:
        landmark.resume(f"./checkpoints/{landmark.modelname}_best.ckpt")
    except FileNotFoundError:
        pass

    landmark.model = landmark.model.to(device)  # move the model parameters to CPU/GPU

    # stage 1 - 3
    for e in range(landmark.cur_epoch, args.stage_epoch):
        landmark.cur_epoch = e + 1
        for loader_index in range(len(landmark.loader_train_sets)):
            landmark.train(loader_index)
            landmark.save(loader_index)
        landmark.val()
        landmark.scheduler.step()

    unfreeze_resnet50_bottom(landmark)
    for e in range(max(args.stage_epoch, landmark.cur_epoch), landmark.tot_epochs):
        landmark.cur_epoch = e + 1
        for loader_index in range(len(landmark.loader_train_sets)):
            landmark.train(loader_index)
            landmark.save(loader_index)
        landmark.val()
        landmark.scheduler.step()


def run(pretrain=True):
    """
       If pretraine is True, we tune All parameters.
       Otehrwise, just train fresh model.
    """

    # Load model
    modelname = args.model_name

    # Visualization
    vis = Visdom(env=modelname)

    # Landmark object
    landmark = Landmark(modelname, load_dataset, vis,
                        pretrained=pretrain,  # determine finetune or not
                        use_stage=False,
                        device=device,
                        lr=args.lr,
                        epochs=args.tot_epochs,
                        batch_size=args.batch_size,
                        optim_params=args.optim_params)

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
            landmark.save(loader_index)
        landmark.val()
        landmark.scheduler.step()


if __name__ == "__main__":
    if args.use_stage:
        finetune_stage()
    else:
        run(args.finetune)
