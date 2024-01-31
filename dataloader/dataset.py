import os
import random
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import os
import torch
from torchvision.transforms import functional as F

from utils.util_functions import *
import numpy as np

import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, args,training):
        self.training = training
        data_dir = args.data_dir
        self.classes = sorted(os.listdir(args.data_dir))
        self.args = args

        self.samples = []
        # self.mask_classes = []
        for i, cls_name in enumerate(self.classes):
            cls_dir = os.path.join(data_dir, cls_name)
            for sample_name in os.listdir(cls_dir):
                sample_path = os.path.join(cls_dir, sample_name)
                self.samples.append((sample_path, i))
        
        self.bounding_box = get_bounding_box_from_roi_file(self.args.roi_info, self.samples[0][0])
        self.transform = transforms.Compose([
           
            #transforms.Resize((300, 300)),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path, label = self.samples[idx]
        sample = Image.open(sample_path).convert('RGB')
        if self.training:
            width, height = sample.size
            # todo. probability of augmentation of image should be for each label
            # eg.
            #if random.random() < self.args.class_augmentation[label]:

            # Randomly augment the image
            if random.random() < 0.5:
                
                # Randomly flip the image horizontally
                if random.random() < self.args.flip:
                    sample = sample.transpose(Image.FLIP_LEFT_RIGHT)
                
                
                # Randomly rotate the image
                if random.random() < self.args.rotate:
                    angle = random.randint(-10, 10)
                    sample = sample.rotate(angle)
                
                
                # Randomly zoom the image
                if random.random() < self.args.zoom:
                    zoom = random.uniform(0.8, 1.2)
                    new_width = int(width * zoom)
                    new_height = int(height * zoom)
                    sample = sample.resize((new_width, new_height))
                    sample = sample.crop((0, 0, width, height))

                # Randomly Noise the image
                if random.random() < self.args.noise:
                    
                    # Convert the PIL Image to a PyTorch tensor
                    to_tensor = transforms.ToTensor()
                    sample = to_tensor(sample)

                    # Generate the noise tensor with the same size as `sample`
                    noise = torch.randn(sample.size()) * 0.1

                    sample = sample + noise
                    sample = sample.clamp(0, 1)

                    # Convert the tensor back to a PIL Image, if necessary
                    to_pil = transforms.ToPILImage()
                    sample = to_pil(sample)
                
                # Randomly shift the image
                if random.random() < self.args.shift:
                    shift_x = random.randint(-10, 10)
                    shift_y = random.randint(-10, 10)
                    sample = sample.transform(sample.size, Image.AFFINE, (1, 0, shift_x, 0, 1, shift_y))
                
                # Randomly squeeze the image
                if random.random() < self.args.squeeze:
                    squeeze = random.uniform(0.8, 1.2)
                    new_width = int(width * squeeze)
                    sample = sample.resize((new_width, height))
                    sample = sample.crop((0, 0, width, height))
                
        #print the shape and dimensions of the image
                    
        print(self.bounding_box)
        # Crop the image
        sample = F.crop(sample, self.bounding_box[1], self.bounding_box[0], self.bounding_box[3] - self.bounding_box[1], self.bounding_box[2] - self.bounding_box[0])
        # Resize the image
        sample = sample.convert('RGB')  # Convert back to RGB
        sample = self.transform(sample)
        # # show the dimensions of the image tensor
        # print("image shape", sample.size)

        # # Assuming sample is your image data
        # sample_np = sample.numpy()  # Convert the tensor to numpy array
        sample = np.transpose(sample, (1, 2, 0))  # Rearrange the dimensions

        plt.imshow(sample)
        plt.show()
        plt.pause(0.001) 
        #wait for enter to be pressed
        input("Press [enter] to continue.")
        plt.close()

        return sample, label


