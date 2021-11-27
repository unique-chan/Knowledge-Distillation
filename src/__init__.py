import torchvision.transforms as transforms


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
