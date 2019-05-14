from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from configuration import CONF
from util import gap_accuracy

PRINT_EVERY = CONF.print_every


class Landmark:
    def __init__(self, model, modelname, loader, vis, device,
                 lr=1e-4, epochs=10, optim_params=None):
        self.device = device
        self.vis = vis
        self.win_train_loss = None
        self.win_val_acc = None

        self.model = model
        self.modelname = modelname
        self.lr = lr
        self.loader_train_sets, self.loader_val, _ = loader()

        # TODO update params_to_update
        self.params_to_update = model.parameters()

        optim_name = optim_params.get('name', 'sdg')
        if optim_name == 'adam':
            weight_decay = optim_params.get('weight_decay', 0)
            self.optimizer = optim.Adam(self.params_to_update, lr=lr, weight_decay=weight_decay)
        elif optim_name == 'sgd':
            momentum = optim_params.get('momentum', 0.9)
            weight_decay = optim_params.get('weight_decay', 5e-4)
            self.optimizer = optim.SGD(self.params_to_update, lr=lr, momentum=momentum,
                                       weight_decay=weight_decay)
        # factor lr by 0.1 in plateau
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', patience=30, cooldown=10,
                                                              verbose=True)

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
                print(f"===> Train on epoch {self.cur_epoch} / {self.tot_epochs} \t|\t "
                      f"loss: {loss.item():.4f} \t|\t batch: [{t}/{self.batch_nums}] \t|\t "
                      f"acc = {100 * float(num_correct) / num_samples:.7f}")

                self.vis.images(x.cpu().data.numpy(), opts=dict(title=f"landmark_id: {y.cpu().numpy()}"))
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

            print(f"===> *Val* on epoch {self.cur_epoch} / {self.tot_epochs} \t|\t "
                  f"acc: {100 * acc:.7f} \t|\t loss: {loss.item():.4f} \t|\t GAP: {gap:.7f}")

            self.vis.images(x.cpu().data.numpy(), opts=dict(title=f"landmark_id: {y.cpu().numpy()}"))
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

    def submit(self):
        pass
