import datetime
from warnings import filterwarnings

import torch

from src.my_utils import parser, util
from __init__ import *
from src import loader, model, iterator

if __name__ == '__main__':
    # Ignore Warning Messages
    filterwarnings('ignore')

    # Random Seeds For Reproducibility
    # util.fix_random_seed()

    # Parser
    my_parser = parser.Parser(mode='train')
    my_args = my_parser.parser.parse_args()

    # Tag
    cur_time = datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    tag_name = f'{my_args.tag}_{cur_time}' if my_args.tag else f'{cur_time}'
    print(f'{tag_name} experiment has been started.')

    # Loader (Train / Valid)
    my_loader = loader.Loader(my_args.dataset_dir, my_args.batch_size,
                              my_args.mean, my_args.std,
                              my_args.auto_mean_std,  # if my_args.auto_mean_std is True,
                                                      # my_args.mean, my_args.std become ignored.
                              my_args.transform_list_name)
    my_train_loader = my_loader.get_loader(mode='train', shuffle=True)
    my_valid_loader = my_loader.get_loader(mode='valid', shuffle=False)

    # Initialization
    my_model = model.model(my_args.network_name, my_loader.num_classes, pretrained=False)
    my_device = 'cpu' if my_args.gpu_index == -1 else f'cuda:{my_args.gpu_index}'
    my_optimizer = get_optimizer(my_model, my_args.lr)                                        # see '__init__.py'
    my_lr_scheduler = get_lr_scheduler(my_optimizer, my_args.lr_step, my_args.lr_step_gamma)  # see '__init__.py'

    # Iterator
    my_iterator = iterator.Iterator(my_model, my_optimizer, my_lr_scheduler, my_loader.num_classes, tag_name,
                                    my_device, my_args.store_weights, my_args.store_loss_acc_log, my_args.store_logits)
    my_iterator.set_loader('train', my_train_loader)
    my_iterator.set_loader('valid', my_valid_loader)

    # Train and valid
    for cur_epoch in range(0, my_args.epochs):
        my_iterator.train(cur_epoch=cur_epoch)
        my_iterator.valid(cur_epoch=cur_epoch)

    if my_args.store_weights:
        my_iterator.store_model()

    print(f'{tag_name} experiment has been done.')
