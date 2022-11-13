import numpy as np
import transforms
from dataloader import MyNyuDataloader

iheight,  iwidth = 480, 640


class NYUDataset(MyNyuDataloader):
    def __init__(self, root, split, modality='rgb'):
        self.split = split
        super(NYUDataset, self).__init__(root, split, modality)
        self.output_size = (224, 224)

    def is_image_file(self, filename):
        if self.split == 'train':
            return (filename.endswith('.h5') and \
                    '00001.h5' not in filename and '00201.h5' not in filename)
        elif self.split == 'val':
            return (filename.endswith('.h5'))
        else:
            raise (RuntimeError("Invalid dataset split:" + "\n"
                                "Supported dataset splits are: train, val"))

    def train_transform(self, rgb, depth):
        s = np.random.uniform(1.0, 1.5)
        depth_np = depth / s
        angle = np.random.uniform(-5.0, 5.0)
        do_flip = np.random.uniform(0.0, 1.0) < 0.5

        transform = transforms.Compose([transforms.Resize(250.0 / iheight),
                                        transforms.Rotate(angle),
                                        transforms.Resize(s),
                                        transforms.CenterCrop((228, 304)),
                                        transforms.HorizontalFlip(do_flip),
                                        transforms.Resize(self.output_size)])
        rgb_np = transform(rgb)
        rgb_np = self.color_jitter(rgb_np)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)
        return rgb_np, depth_np

    def val_transform(self, rbg, depth):
        depth_np = depth
        transform = transforms.Compose([transforms.Resize(250.0 / iheight),
                                        transforms.CenterCrop((228, 304)),
                                        transforms.Resize(self.output_size)])
        rgb_np = transform(rbg)
        rgb_np = np.asfarray(rgb_np, dtype='float') / 255
        depth_np = transform(depth_np)

        return rgb_np, depth_np
