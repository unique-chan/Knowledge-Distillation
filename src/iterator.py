import csv
import os

import torch
import torch.nn as nn

from src.my_utils import util
from __init__ import *

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

LOSS_ACC_STATE_FIELDS = ['epoch', 'train_loss', 'valid_loss',
                         'train_top1_acc', 'train_top5_acc', 'valid_top1_acc', 'valid_top5_acc']
LOGITS_STATE_FIELDS = ['epoch', 'output_distribution', 'classification_result']


class Iterator:
    def __init__(self, model, optimizer, lr_scheduler, num_classes, tag_name,
                 device='cpu', store_weights=False, store_loss_acc_log=False, store_logits=False):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.num_classes = num_classes
        self.device = device  # 'cpu', 'cuda:0', ...
        self.model.to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.loader = {'train': None, 'valid': None, 'test': None}
        self.store_weights = store_weights
        self.store_loss_acc_log = store_loss_acc_log
        self.store_logits = store_logits
        self.tag_name = tag_name
        self.best_valid_acc_state = {'top1_acc': 0, 'top5_acc': 0}
        if store_weights or store_loss_acc_log or store_logits:
            os.makedirs(f'{LOG_DIR}/{tag_name}', exist_ok=True)
        if store_weights:
            # [GOAL] store the best validation model during training.
            self.best_model_state_path = f'{LOG_DIR}/{tag_name}/{tag_name}_valid_best.pt'
            self.best_model_state_dict = self.model.state_dict()
        if store_loss_acc_log:
            # [GOAL] store train/valid loss & acc per each epoch during training.
            self.loss_acc_state = {field_name: 0 for field_name in LOSS_ACC_STATE_FIELDS}
            self.log_loss_acc_csv_path = f'{LOG_DIR}/{tag_name}/{tag_name}.csv'
            self.log_loss_acc_csv_writer = csv.DictWriter(open(self.log_loss_acc_csv_path, 'w', newline=NEWLINE),
                                                          fieldnames=LOSS_ACC_STATE_FIELDS)
            self.log_loss_acc_csv_writer.writeheader()
        if store_logits:
            # [GOAL] store output distributions per each epoch for all images in the current experiment.
            self.logits_root_path = f'{LOG_DIR}/{self.tag_name}/logits'
            self.logits_csv_writers = {}  # key: 'img_path', value: csv_writer for corresponding key

    def set_loader(self, mode, loader):
        self.loader[mode] = loader

    def one_epoch(self, mode, cur_epoch):
        loader = self.loader[mode]
        meter = {'loss': util.Meter(), 'top1_acc': util.Meter(), 'top5_acc': util.Meter()}
        assert loader, f"No loader['{mode}'] exists. Pass the loader to the Iterator via set_loader()."
        tqdm_loader = tqdm.tqdm(loader, mininterval=0.1) if bool_tqdm else loader
        classification_results = []
        output_distributions = []
        img_paths = []
        for (img_path, x, y) in tqdm_loader:
            x, y = x.to(self.device), y.to(self.device)
            # predict
            output_distribution = self.model(x)
            classification_result, [top1_acc, top5_acc] = \
                Iterator.__get_final_classification_results_and_topk_acc__(output_distribution, y, top_k=(1, 5))
            meter['top1_acc'].update(top1_acc.item(), k=output_distribution.size(0))
            meter['top5_acc'].update(top5_acc.item(), k=output_distribution.size(0))
            # calculate loss
            if mode in ['train', 'valid']:
                loss = self.criterion(output_distribution, y)
                meter['loss'].update(loss.item(), k=output_distribution.size(0))
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
            # accumulate the prediction results
            if self.store_logits:
                img_paths.extend(img_path)
                classification_results.extend(torch.flatten(classification_result).tolist())
                output_distributions.extend([logit.tolist() for logit in output_distribution.cpu().detach().numpy()])
            if bool_tqdm:
                tqdm_loader.set_description(f'{mode.upper()} | {cur_epoch + 1:>5d} | {log_msg}')
        return meter['loss'].avg, meter['top1_acc'].avg * 100., meter['top5_acc'].avg * 100., \
               img_paths, classification_results, output_distributions

    def train(self, cur_epoch):
        mode = 'train'
        self.model.train()
        loss, top1_acc, top5_acc, img_paths, classification_results, output_distributions = \
            self.one_epoch(mode=mode, cur_epoch=cur_epoch)
        self.lr_scheduler.step()
        if self.store_loss_acc_log:
            self.__update_loss_acc_state(mode, cur_epoch, loss, top1_acc, top5_acc)
        if self.store_logits:
            self.__write_csv_logits(mode, cur_epoch, img_paths, classification_results, output_distributions)

    def valid(self, cur_epoch):
        mode = 'valid'
        self.model.eval()
        with torch.no_grad():
            loss, top1_acc, top5_acc, img_paths, classification_results, output_distributions = \
                self.one_epoch(mode=mode, cur_epoch=cur_epoch)
        self.__update_best_valid_acc_state(top1_acc, top5_acc)
        if self.store_weights:
            self.best_model_state_dict = self.model.state_dict()
        if self.store_loss_acc_log:
            self.__update_loss_acc_state(mode, cur_epoch, loss, top1_acc, top5_acc)
            self.__write_csv_log_loss_acc()
        if self.store_logits:
            self.__write_csv_logits(mode, cur_epoch, img_paths, classification_results, output_distributions)

    def test(self):
        mode = 'test'
        self.model.eval()
        with torch.no_grad():
            _, top1_acc, top5_acc, img_paths, classification_results, output_distributions = \
                self.one_epoch(mode=mode, cur_epoch=-1)
        if self.store_logits:
            self.__write_csv_logits(mode, -1, img_paths, classification_results, output_distributions)

    def store_model(self):
        torch.save(self.best_model_state_dict, self.best_model_state_path)

    @classmethod
    def __get_final_classification_results_and_topk_acc__(cls, out, gt, top_k=(1, 5)):
        _, prediction = out.topk(k=max(top_k), dim=1, largest=True, sorted=True)
        prediction = prediction.t()
        correct = prediction.eq(gt.view(1, -1).expand_as(prediction))
        top_k_acc_list = [correct[:k].reshape(-1).float().sum(0, keepdim=True) for k in top_k]
        _, top_1_prediction = out.topk(k=1, dim=1, largest=True, sorted=True)
        return top_1_prediction, top_k_acc_list  # sum of correct predictions (top_1, top_k)

    def __update_best_valid_acc_state(self, top1_acc, top5_acc):
        if top1_acc > self.best_valid_acc_state['top1_acc'] or \
                (top1_acc == self.best_valid_acc_state['top1_acc'] and
                 top5_acc > self.best_valid_acc_state['top5_acc']):
            self.best_valid_acc_state['top1_acc'] = top1_acc
            self.best_valid_acc_state['top5_acc'] = top5_acc

    def __update_loss_acc_state(self, mode, epoch, loss, top1_acc, top5_acc):
        self.loss_acc_state['epoch'] = epoch
        self.loss_acc_state[f'{mode}_loss'] = loss
        self.loss_acc_state[f'{mode}_top1_acc'] = top1_acc
        self.loss_acc_state[f'{mode}_top5_acc'] = top5_acc

    def __write_csv_log_loss_acc(self):
        with open(self.log_loss_acc_csv_path, 'a') as f:
            self.log_loss_acc_csv_writer.writerow(self.loss_acc_state)
            f.flush()

    def __write_csv_logits(self, mode, cur_epoch, img_paths, classification_results, output_distributions):
        # assert len(img_paths) == len(classification_results) == len(output_distributions)
        zips = zip(img_paths, classification_results, output_distributions)
        sep = os.sep  # '/': linux, '\': windows
        for img_path, classification_result, output_distribution in zips:
            class_name, file_name = img_path.split(sep)[-2], img_path.split(sep)[-1]
            root_path = f'{self.logits_root_path}/{mode}/{class_name}/{file_name}'
            if not os.path.isdir(root_path):
                os.makedirs(root_path, exist_ok=True)
                csv_path = f'{root_path}/logits.csv'
                csv_writer = csv.DictWriter(open(csv_path, 'w', newline=NEWLINE), fieldnames=LOGITS_STATE_FIELDS)
                csv_writer.writeheader()
                self.logits_csv_writers[f'{mode}/{class_name}/{file_name}'] = csv_writer
            with open(self.log_loss_acc_csv_path, 'a') as f:
                self.logits_csv_writers[f'{mode}/{class_name}/{file_name}'].writerow({
                    'epoch': cur_epoch,
                    'output_distribution': output_distribution,
                    'classification_result': classification_result
                })
                f.flush()
