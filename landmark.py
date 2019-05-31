from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from configuration import CONF
from resnet import resnet50
from utils.util import gap_accuracy

from squeezenet import squeezenet1_1
from mobilenet import mobilenet_v2

PRINT_EVERY = CONF.print_every
LEARNING_RATE = 1e-2

class Landmark:
    def __init__(self, modelname, loader, device, batch_size, pretrained=True, use_stage=False,
                 lr=LEARNING_RATE, epochs=20, optim_params={}, params_to_update=None):
        self.device = device
        self.win_train_loss = None
        self.win_val_acc = None

        self.lr = lr
        self.batch_size = batch_size
        self.loader_train_sets, self.loader_val, _, num_classes = loader(batch_size)

        #self.model = resnet50(pretrained=pretrained, num_classes=num_classes)
        self.model = mobilenet_v2(pretrained=pretrained, num_classes=num_classes)

        self.modelname = modelname

        if params_to_update is None:
            self.params_to_update = self.model.parameters()
            if pretrained and use_stage:
                self.params_to_update = self.model.fc.parameters()
        else:
            self.params_to_update = params_to_update

        optim_name = optim_params.get('name', 'adam')

        if optim_name == 'adam':
            weight_decay = optim_params.get('weight_decay', 0)
            self.optimizer = optim.Adam(self.params_to_update, lr=lr, weight_decay=weight_decay, amsgrad=True)
        elif optim_name == 'sgd':
            momentum = optim_params.get('momentum', 0.9)
            weight_decay = optim_params.get('weight_decay', 1e-4)
            self.optimizer = optim.SGD(self.params_to_update, lr=lr, momentum=momentum,
                                       weight_decay=weight_decay)
        # factor lr by 0.1 in plateau
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.5)

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
            savename = f"./checkpoints/{self.modelname}_best.ckpt"
            print(f"===> ===> Saving model in \"{savename}\" with acc {100 * acc:.7f} "
                  f"in epoch {self.cur_epoch} / {self.tot_epochs}...")
            torch.save(state, savename)
            self.best_acc = acc

    def resume(self, ckpt_path):
        if not Path(ckpt_path).exists():
            print(f"\"{ckpt_path}\" not found...")
        checkpoint = torch.load(ckpt_path)
        self.model.load_state_dict(checkpoint['model'])
        self.best_acc = checkpoint['acc']
        self.cur_epoch = checkpoint['epoch']
        print(f"*** Resume checkpoint from \"{self.modelname}\" "
              f"with acc {100 * self.best_acc:.7f} in epoch {self.cur_epoch} / {self.tot_epochs}...")

    def save(self, loader_index):
        state = {
            'model': self.model.state_dict(),
            'acc'  : None,
            'epoch': self.cur_epoch,
        }
        if not Path('checkpoints').exists():
            Path('checkpoints').mkdir()
        savename = f"./checkpoints/{self.modelname}_newest.ckpt"
        print(f"===> ===> Saving model in \"{savename}\" by {loader_index}-th training set "
              f"in epoch {self.cur_epoch} / {self.tot_epochs}...")
        torch.save(state, savename)

    def submit(self):
        pass
