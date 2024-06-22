import os
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import random
import cv2
import keras
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
from torch.autograd import Variable

def find_classes(root):
    classes = [d for d in os.listdir(root)]
    classes.sort()
    class_to_idex = {classes[i]: i for i in range(len(classes))}
    index_to_class = {i: classes[i] for i in range(len(classes))}
    return classes, class_to_idex, index_to_class

def make_dataset(root):
    images = []

    cnames = os.listdir(root)
    for cname in cnames:
        c_path = os.path.join(root, cname)
        fnames = os.listdir(c_path)
        for fname in fnames:
            path = os.path.join(c_path, fname)
            images.append(path)

    return images


class TripleDataset(data.Dataset):
    def __init__(self, photo_root, sketch_root, batch_size=32):
        super(TripleDataset, self).__init__()
        self.tranform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        classes, class_to_idx, idx_to_class = find_classes(photo_root)

        self.photo_root = photo_root
        self.sketch_root = sketch_root
        
        self.anchor_images = sorted(make_dataset(self.sketch_root))
        
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = idx_to_class

        self.batch_size = batch_size
        self.len = len(self.anchor_images)
        self.num_batches = len(self.anchor_images) // self.batch_size
        
    def __getitem__(self, index):
<<<<<<< HEAD
        anchor_data = self.anchor_images[index]
        pos_data, label = self._getrelate_photo(anchor_data)
        neg_data = self._getneg_photo(anchor_data)
        
#         print(anchor_data, pos_data, neg_data)
        po = Image.open(pos_data).convert('RGB')
        an = Image.open(anchor_data).convert('RGB')
        ne = Image.open(neg_data).convert('RGB')
        
#         print(an.size, po.size, ne.size)
        A = self.tranform(an)
        P = self.tranform(po)
        L = str(label)
        N = self.tranform(ne)
#         print(A.mode, P.mode, N.mode)
#         L1 = label1
        
        return {'A': A, 'P': P, 'N': N, 'L': L}
    
=======

        photo_path = self.photo_paths[index]
        sketch_path, label = self._getrelate_sketch(photo_path)

        if label == 0:
            r = list(range(label+1,125))
        else:
            r = list(range(0,label))+list(range(label+1,125))
        i = random.choice(r) #class

        neg_path, label1 = self._getneg_photo(i)
        # print(photo_path, sketch_path, neg_path, label, label1)

        photo = Image.open(photo_path).convert('RGB')
        sketch = Image.open(sketch_path).convert('RGB')
        neg = Image.open(neg_path).convert('RGB')
        
        P = self.tranform(photo)
        A = self.tranform(sketch)
        L = label
        N = self.tranform(neg)
        L1 = label1

        return {'A': A, 'P': P, 'N': N, 'L': L, 'L1': L1}
        # return A, P, N, L, L1

>>>>>>> 7cc04772d4a67870dfbb5a6fac3aa4253ba46658
    def __len__(self):
        return self.len
    def __iter__(self):
        for index in range(len(self)):
            yield self[index]
    
    def _getrelate_photo(self, anchor_filename):
        # paths = anchor_filename.split('/')
        paths = os.path.normpath(anchor_filename).split(os.path.sep)
        fname = paths[-1].split('-')[0]
        cname = paths[-2]

        label = self.class_to_idx[cname]
        
        pos = '0'
        photos = os.listdir(os.path.join(self.photo_root, cname))

        for photo_name in photos:
            name = photo_name.split('.')[0]
            if  name == fname:
                pos = os.path.join(self.photo_root, cname, photo_name)
        return pos, label
    
    def _getneg_photo(self, anchor_filename):
        # paths = anchor_filename.split('/')
        paths = os.path.normpath(anchor_filename).split(os.path.sep)
        fname = paths[-1].split('-')[0]
        cname = paths[-2]

<<<<<<< HEAD
=======
        ran = random.randint(0, 7)

        neg_img = paths[ran]
>>>>>>> 7cc04772d4a67870dfbb5a6fac3aa4253ba46658
        label = self.class_to_idx[cname]
        if label == 0:
            r = list(range(label+1,10))
        else:
            r = list(range(0,label))+list(range(label+1,10))
        i = random.choice(r) #class
        negative_class = self.idx_to_class[i]
        # Kiểm tra xem thư mục của lớp âm có hình ảnh hay không
        negative_images_path = os.path.join(self.photo_root, negative_class)
        if len(os.listdir(negative_images_path)) == 0:
            # Nếu thư mục trống, chọn một hình ảnh khác
            return self._getneg_photo(anchor_filename)

<<<<<<< HEAD
        negative_image = random.choice(os.listdir(negative_images_path))  
        return os.path.join(self.photo_root,negative_class, negative_image)
=======
        return neg_img, label
>>>>>>> 7cc04772d4a67870dfbb5a6fac3aa4253ba46658
