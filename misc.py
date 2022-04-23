
from torchvision.datasets import ImageFolder
from torchvision import transforms 
from torch.utils import data



def loader(data_dir, mode, img_crop, img_size, data_name, batch_size, workers):
    T = []
    if mode == 'train':
        T.append(transforms.RandomHorizontalFlip())
    T.append(transforms.CenterCrop(img_crop))
    T.append(transforms.Resize(img_size))
    T.append(transforms.ToTensor())
    T.append(transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    T = transforms.Compose(T)

    data_name = ImageFolder(data_dir, T)

    loader = data.DataLoader(dataset=data_name, 
                            batch_size=batch_size, 
                            shuffle=(mode=='train'), 
                            num_workers=workers)

    return loader
