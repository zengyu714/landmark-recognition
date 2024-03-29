from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from config import Config
from models.resnet import resnet50, resnext50_32x4d
from models.se_resnet import se_resnet50
from models.seatt import *
from utils.util import gap_accuracy

PRINT_EVERY = Config().print_every


class Landmark:
    def __init__(self, modelname, nickname, loader, vis, device, batch_size, input_size,
                 pretrained=True, use_stage=False, step_size=3, lr=1e-4, epochs=10, optim_params={}):
        """
        Args:
            - modelname (str): The name of the model.
                Currently we support: ['seatt154', 'seatt_base56', 'seatt_base92', 'seatt_resnext50_32x4d',
                                       'seatt_resnet50', 'se_resnet50', 'resnet50', 'seatt_resnext50_base']
            - nickname (str): The name of this experiment. E.g., 'resnet50_foo', 'bar_seatt154'...
            - vis (object): Handle of the visdom object
            - device (int): The index of the device.
            - batch_size (int): Mini batch size for a forward.
            - input_size (tuple): Input image size.
            - pretrained (bool):  If `True`, use pretrained weights fro ImageNet.
            - use_stage (bool): If `True`, use three 3 stage finetune strategy.
            - step_size (int): Step size (epoch) to factor the learning rate. `-1` means no decay
            - lr (float): Learning rate.
            - epochs (int): Number of total training epochs.
            - optim_params (dict): The configuration of the optimizer.
        """
        self.device = device
        self.vis = vis
        self.win_train_loss = None
        self.win_val_acc = None
        self.nickname = nickname

        self.use_stage = use_stage
        self.pretrained = pretrained

        self.lr = lr
        self.batch_size = batch_size
        self.loader_train_sets, self.loader_val, _, num_classes = loader(input_size, batch_size)

        try:
            self.model = eval(modelname)(pretrained=pretrained, num_classes=num_classes)
        except NameError:
            print(f"No support for {modelname}")
            exit(1)

        if use_stage:
            assert pretrained == True, "We need pretrained weights to use stage finetune strategy"
            self.params_to_update = self.model.fc.parameters()
        else:
            self.params_to_update = self.model.parameters()

        optim_name = optim_params.get('name', 'adam')

        if optim_name == 'adam':
            weight_decay = optim_params.get('weight_decay', 0)
            self.optimizer = optim.Adam(self.params_to_update, lr=lr, weight_decay=weight_decay, amsgrad=True)
        elif optim_name == 'sgd':
            momentum = optim_params.get('momentum', 0.9)
            weight_decay = optim_params.get('weight_decay', 5e-4)
            self.optimizer = optim.SGD(self.params_to_update, lr=lr, momentum=momentum, weight_decay=weight_decay)
        # factor lr by 0.5 in plateau
        self.scheduler = None
        if step_size > 1:
            self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=0.5)

        self.batch_nums = len(self.loader_val)
        self.best_acc = 0
        self.tot_epochs = epochs
        self.cur_epoch = 0

    def train(self, loader_index):
        num_correct = 0
        num_samples = 0
        self.model.train()  # put model to training mode

        for t, sample in enumerate(self.loader_train_sets[loader_index]):
            x = sample['image'].to(device=self.device, dtype=torch.float32)  # move to device, e.g. GPU
            y = sample['landmark_id'].to(device=self.device, dtype=torch.long)

            scores = self.model(x)
            loss = F.cross_entropy(scores, y)

            self.optimizer.zero_grad()

            loss.backward()
            self.optimizer.step()

            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)

            if t % PRINT_EVERY == 0:
                acc = 100 * float(num_correct) / num_samples
                now = datetime.now().strftime('%m-%d %H:%M:%S')
                print(f"[{now}] Train on epoch {self.cur_epoch} / {self.tot_epochs}  |  "
                      f"loss: {loss.item():.4f}  |  batch: [{t}/{self.batch_nums} ({loader_index}/"
                      f"{len(self.loader_train_sets)})]  |  "
                      f"acc = {acc:.7f}")

                self.vis.images(x.cpu().data.numpy() * 255, opts=dict(title=f"landmark_id: {y.cpu().numpy()}"))
                if self.win_train_loss is None:
                    self.win_train_loss = self.vis.line(
                            X=np.array([self.cur_epoch + t / len(self.loader_train_sets[0])]),
                            Y=np.array([loss.item()]))
                else:
                    self.vis.line(
                            X=np.array([self.cur_epoch + t / len(self.loader_train_sets[0])]),
                            Y=np.array([loss.item()]), win=self.win_train_loss, name='training loss', update='append')

    def val(self):
        num_correct = 0
        num_samples = 0
        tot_loss = 0

        # save for computing GAP
        preds_all = []
        probs_all = []
        trues_all = []

        self.model.eval()  # set model to evaluation mode

        with torch.no_grad():
            for sample in self.loader_val:
                x = sample['image'].to(device=self.device, dtype=torch.float32)  # move to device, e.g. GPU
                y = sample['landmark_id'].to(device=self.device, dtype=torch.long)
                scores = self.model(x)
                tot_loss += F.cross_entropy(scores, y)
                softmax = F.softmax(scores, dim=1)

                probs, preds = softmax.max(1)
                num_correct += (preds == y).sum()
                num_samples += preds.size(0)

                preds_all.append(preds)
                probs_all.append(probs)
                trues_all.append(y)

            acc = float(num_correct) / num_samples
            loss = tot_loss / num_samples
            # compute gap
            args = [torch.cat(i).cpu().numpy() for i in [preds_all, probs_all, trues_all]]
            gap = gap_accuracy(*args, return_df=False)
            now = datetime.now().strftime('%m-%d %H:%M:%S')

            print(f"[{now}] *Val* on epoch {self.cur_epoch} / {self.tot_epochs}  |  "
                  f"acc: {100 * acc:.7f}  |  loss: {loss.item():.4f}  |  GAP: {gap:.7f}")

            self.vis.images(x.cpu().data.numpy() * 255, opts=dict(title=f"landmark_id: {y.cpu().numpy()}"))
            if self.win_val_acc is None:
                self.win_val_acc = self.vis.line(
                        X=np.column_stack([self.cur_epoch] * 2),
                        Y=np.column_stack([loss.item(), acc]))
            else:
                self.vis.line(
                        X=np.column_stack([self.cur_epoch] * 2),
                        Y=np.column_stack([loss.item(), acc]), win=self.win_val_acc, name='validation loss / acc',
                        update='append')

        if acc > self.best_acc:
            state = {
                'model': self.model.state_dict(),
                'acc'  : acc,
                'epoch': self.cur_epoch,
            }
            if not Path('checkpoints').exists():
                Path('checkpoints').mkdir()
            savename = f"./checkpoints/{self.nickname}_best.ckpt"
            print(f"===> ===> Saving model in \"{savename}\" with acc {100 * acc:.7f} "
                  f"in epoch {self.cur_epoch} / {self.tot_epochs}...")
            torch.save(state, savename)
            self.best_acc = acc

    def resume(self, ckpt_path):
        if not Path(ckpt_path).exists():
            print(f"\"{ckpt_path}\" not found...")
        checkpoint = torch.load(ckpt_path, map_location=lambda storage, loc: storage.cuda())
        self.model.load_state_dict(checkpoint['model'])
        self.best_acc = checkpoint['acc']
        self.cur_epoch = checkpoint['epoch']
        try:
            print(f"*** Resume checkpoint from best \"{self.nickname}\" "
                  f"with acc {100 * self.best_acc:.7f} in epoch {self.cur_epoch} / {self.tot_epochs}...")
        except TypeError:
            # If model doesn't have `best_acc` and `cur_epoch` since resumed from newest
            self.best_acc = 0.0
            print(f"*** Resume checkpoint from newest \"{self.nickname}\"...")

    def save(self, loader_index):
        state = {
            'model': self.model.state_dict(),
            'acc'  : None,
            'epoch': self.cur_epoch,
        }
        if not Path('checkpoints').exists():
            Path('checkpoints').mkdir()
        savename = f"./checkpoints/{self.nickname}_newest.ckpt"
        print(f"===> ===> Saving model in \"{savename}\" by {loader_index}-th training set "
              f"in epoch {self.cur_epoch} / {self.tot_epochs}...")
        torch.save(state, savename)
