import torchvision.transforms as transforms
import torch.optim as optim
import sys

sys.path.append('src')  # Do not remove this code!


def get_manual_transform_list(mode, transform_list_name, mean, std):
    if mode == 'train':
        transform_list_dir = {
            'CIFAR': [
                transforms.RandomCrop((32, 32), padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ],  # you can add your own manual transform list!
        }
    else:  # for 'valid' and 'test'
        transform_list_dir = {
            'CIFAR': [
                transforms.ToTensor(),
            ],  # you can add your own manual transform list!
        }

    transform_list = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if transform_list_dir.get(transform_list_name):
        transform_list = transform_list_dir[transform_list_name] + [transforms.Normalize(mean, std)]
    return transform_list


def get_optimizer(model, lr):
    return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)


def get_lr_scheduler(optimizer, lr_step, lr_step_gamma):
    return optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_step, gamma=lr_step_gamma)
