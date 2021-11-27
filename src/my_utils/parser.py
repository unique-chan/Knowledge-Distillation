import argparse


class Parser:
    def __init__(self, mode):
        self.parser = argparse.ArgumentParser(description='Pytorch Image Classification (github.com/unique-chan)')
        if mode == 'train':
            self.add_arguments_for_train()
        elif mode == 'test':
            self.add_arguments_for_test()
        self.add_default_arguments()

    def add_default_arguments(self):
        self.parser.add_argument('--network_name', type=str, help='network name')
        self.parser.add_argument('--dataset_dir', type=str, help='dataset path')
        self.parser.add_argument('--batch_size', default=128, type=int, help='batch_size (default: 128)')
        self.parser.add_argument('--mean', default="()")
        self.parser.add_argument('--std', default="()")
        self.parser.add_argument('--gpu_index', default=0, type=int,
                                 help="[gpu_index = -1]: cpu, [gpu_index >= 0]: gpu")

    def add_arguments_for_train(self):
        self.parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate (default: 0.1)')
        self.parser.add_argument('--epochs', default=1, type=int, help='epochs (default: 1)')
        self.parser.add_argument('--lr_step', default="[100, 150]", type=str,
                                 help='learning rate step decay milestones (default: [100, 150])')
        self.parser.add_argument('--lr_step_gamma', default=0.1, type=float,
                                 help='learning rate step decay gamma (default: 0.1)')

    def add_arguments_for_test(self):
        pass
