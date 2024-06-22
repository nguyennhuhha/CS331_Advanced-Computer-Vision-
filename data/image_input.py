import os
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


def make_dataset(root):
    images = []

    cnames = os.listdir(root)
    for cname in cnames:
        c_path = os.path.join(root, cname)
        if os.path.isdir(c_path):
            fnames = os.listdir(c_path)
            for fname in fnames:
                path = os.path.join(c_path, fname)
                images.append(path)

    return images


class ImageDataset(data.Dataset):
    def __init__(self, image_root):
        self.tranform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
<<<<<<< HEAD
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
=======
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
>>>>>>> 7cc04772d4a67870dfbb5a6fac3aa4253ba46658
        ])

        self.image_root = image_root
        self.image_paths = sorted(make_dataset(self.image_root))

        self.len = len(self.image_paths)

    def __getitem__(self, index):

        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        image = self.tranform(image)

<<<<<<< HEAD
        image_path = os.path.normpath(image_path)  # Chuẩn hóa đường dẫn
        image_path = image_path.split(os.path.sep)  # Tách đường dẫn với separator của hệ điều hành
        # image_path = image_path.split('/')
        # print(image_path)
=======
        image_path = image_path.split('\\')
>>>>>>> 7cc04772d4a67870dfbb5a6fac3aa4253ba46658
        cname = image_path[-2]
        fname = image_path[-1]
        # print(cname)
        # print(fname)

<<<<<<< HEAD
        # name = cname + '/' + fname
        name = os.path.join(cname, fname)  # Sử dụng os.path.join để tạo đường dẫn chuẩn
        # print(name)

=======
        name = cname + '/' + fname
>>>>>>> 7cc04772d4a67870dfbb5a6fac3aa4253ba46658
        return {'I': image, 'N': name}

    def __len__(self):
        return self.len

