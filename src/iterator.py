import csv

import torch
import torch.nn as nn

from src.my_utils import util

try:
    bool_tqdm = True
    import tqdm
except ImportError:
    bool_tqdm = False
    print('[Warning] Try to install tqdm for progress bar.')

try:
    bool_tb = True
    import tensorboard
except ImportError:
    bool_tb = False
    print('[Warning] Try to install tensorboard for checking the status of learning.')

fieldnames = ['epoch', 'train_loss', 'valid_loss',
              'train_top1_acc', 'train_top5_acc', 'valid_top1_acc', 'valid_top5_acc']


class Iterator:
    def __init__(self, model, optimizer, lr_scheduler, num_classes, tag_name,
                 device='cpu', store=False):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_classes = num_classes
        self.device = device  # 'cpu', 'cuda:0', ...
        self.model.to(device)

        self.csv_writer = csv.DictWriter(open(f'{tag_name}.csv', 'w', newline=''), fieldnames=fieldnames)
        self.csv_writer.writeheader()

        self.criterion = nn.CrossEntropyLoss()
        self.loader = {'train': None, 'valid': None, 'test': None}

        self.log_loss = {'train': [], 'valid': []}
        self.log_top1_acc, self.log_top5_acc = {'train': [], 'valid': []}, {'train': [], 'valid': []}
        self.log_predictions = {'train': [], 'valid': [], 'test': []}

        self.state = {'epoch': 0,
                      'train_loss': 0, 'valid_loss': 0,
                      'train_top1_acc': 0, 'train_top5_acc': 0,
                      'valid_top1_acc': 0, 'valid_top5_acc': 0}

        self.best_model_state_dict = self.model.state_dict() if store else None

    def set_loader(self, mode, loader):
        self.loader[mode] = loader

    def one_epoch(self, mode, cur_epoch):
        loader = self.loader[mode]
        meter = {'loss': util.Meter(), 'top1_acc': util.Meter(), 'top5_acc': util.Meter()}
        assert loader, f"No loader['{mode}'] exists. Pass the loader to the Iterator via set_loader()."
        tqdm_loader = tqdm.tqdm(loader, mininterval=0.1) if bool_tqdm else loader
        predictions = []
        for (x, y) in tqdm_loader:
            x, y = x.to(self.device), y.to(self.device)
            # predict
            output_distributions = self.model(x)
            prediction, [top1_acc, top5_acc] = \
                Iterator.__get_final_prediction_and_topk_acc__(output_distributions, y, top_k=(1, 5))
            meter['top1_acc'].update(top1_acc.item(), k=output_distributions.size(0))
            meter['top5_acc'].update(top5_acc.item(), k=output_distributions.size(0))

            # calculate loss
            if mode in ['train', 'valid']:
                loss = self.criterion(output_distributions, y)
                meter['loss'].update(loss.item(), k=output_distributions.size(0))
                log_msg = f"Loss: {meter['loss'].avg:.3f} | Acc: (top1) {meter['top1_acc'].avg * 100.:.2f}% " \
                          f"(top5) {meter['top5_acc'].avg * 100.:.2f}% "
                # update
                if mode == 'train':
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()
            else:
                log_msg = f"Acc: (top1) {meter['top1_acc'].avg * 100.:.2f}% (top5) {meter['top5_acc'].avg * 100.:.2f}%"

            if bool_tqdm:
                tqdm_loader.set_description(f'{mode.upper()} | {cur_epoch + 1:>5d} | {log_msg}')

            predictions.extend(torch.flatten(prediction).tolist())  # accumulate the prediction results
        return meter['loss'].avg, meter['top1_acc'].avg * 100., meter['top5_acc'].avg * 100., predictions

    def train(self, cur_epoch):
        mode = 'train'
        self.model.train()
        loss, top1_acc, top5_acc, predictions = self.one_epoch(mode=mode, cur_epoch=cur_epoch)
        self.__state_update(mode, cur_epoch, loss, top1_acc, top5_acc)
        self.__log_update(mode, loss, top1_acc, top5_acc, predictions)
        self.lr_scheduler.step()

    def valid(self, cur_epoch):
        mode = 'valid'
        self.model.eval()
        with torch.no_grad():
            loss, top1_acc, top5_acc, predictions = self.one_epoch(mode=mode, cur_epoch=cur_epoch)
            self.__state_update(mode, cur_epoch, loss, top1_acc, top5_acc)
            if self.log_top1_acc[mode]:
                if top1_acc > max(self.log_top1_acc[mode]) or \
                        (top1_acc == max(self.log_top1_acc[mode]) and top5_acc > max(self.log_top5_acc[mode])):
                    self.best_model_state_dict = self.model.state_dict()  # store best model
            self.__log_update(mode, loss, top1_acc, top5_acc, predictions)

    def test(self):
        mode = 'test'
        self.model.eval()
        with torch.no_grad():
            _, top1_acc, top5_acc, predictions = self.one_epoch(mode=mode, cur_epoch=-1)
            self.log_predictions[mode].append(predictions)

    @classmethod
    def __get_final_prediction_and_topk_acc__(cls, out, gt, top_k=(1, 5)):
        _, prediction = out.topk(k=max(top_k), dim=1, largest=True, sorted=True)
        prediction = prediction.t()
        correct = prediction.eq(gt.view(1, -1).expand_as(prediction))
        top_k_acc_list = [correct[:k].reshape(-1).float().sum(0, keepdim=True) for k in top_k]
        _, top_1_prediction = out.topk(k=1, dim=1, largest=True, sorted=True)
        return top_1_prediction, top_k_acc_list  # sum of correct predictions (top_1, top_k)

    def __log_update(self, mode, loss, top1_acc, top5_acc, predictions):
        self.log_loss[mode].append(loss)
        self.log_top1_acc[mode].append(top1_acc)
        self.log_top5_acc[mode].append(top5_acc)
        self.log_predictions[mode].append(predictions)

    def __state_update(self, mode, epoch, loss, top1_acc, top5_acc):
        self.state['epoch'] = epoch
        self.state[f'{mode}_loss'] = loss
        self.state[f'{mode}_top1_acc'] = top1_acc
        self.state[f'{mode}_top5_acc'] = top5_acc
