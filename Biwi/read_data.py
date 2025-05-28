import os.path

import torch
import numpy as np
from sklearn.preprocessing import StandardScaler
import random
from PIL import Image
import math


def connect_image_to_txt(image_file):
    return f'{str(image_file)[:-7]}pose.txt'
def read_bin(filename):
    return np.fromfile(filename)

def make_dataset(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], int(val.split()[1])) for val in image_list]
    return images


def make_dataset_r(image_list, labels):
    if labels:
        len_ = len(image_list)
        images = [(image_list[i].strip(), labels[i, :]) for i in range(len_)]
    else:
        if len(image_list[0].split()) > 2:
            images = [(val.split()[0], np.array([float(la) for la in val.split()[1:]])) for val in image_list]
        else:
            images = [(val.split()[0], float(val.split()[1])) for val in image_list]
    return images

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def pil_loader1(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('L')

def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    # else:
    return pil_loader(path)

def default_loader1(path):
    # from torchvision import get_image_backend
    # if get_image_backend() == 'accimage':
    #    return accimage_loader(path)
    # else:
    return pil_loader1(path)

def rotation_matrix_to_euler_angles(R):
    """
    Converts a 3x3 rotation matrix (R) to Euler angles (Z-Y-X convention).
    """
    assert(R.shape == (3, 3))

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z]) * 180/np.pi # roll, pitch, yaw

def make_dataset_file(root):
    F_index = list(range(1, 7)) + [15] + [18] + [21]
    image_root = os.path.join(root,'faces_0')
    bin_root = os.path.join(root,'db_annotations')
    mf = open('male.txt','w')
    ff = open('fmale.txt', 'w')
    test_mf = open('male_test.txt', 'w')
    test_ff = open('fmale_test.txt', 'w')
    for person_id in os.listdir(bin_root):
        img_id_dir = os.path.join(image_root,person_id)
        print(img_id_dir)
        if int(person_id) in F_index:
            train_f = ff
            test_f = test_ff
        else:
            train_f = mf
            test_f = test_mf
        id_dir = os.listdir(img_id_dir)
        imagenum = (len(id_dir)-1)//3
        randomtest = random.sample(list(range(imagenum)), int(0.1*imagenum))
        print(randomtest)
        print(len(randomtest))
        test_num = 0
        for i,imagepath in enumerate(os.listdir(img_id_dir)):
            if imagepath[-3:]!='png': continue
            ind = i//3
            if ind in randomtest:
                test_num+=1
                f = test_f
            else:
                f = train_f
            posepath = imagepath[:-7]+'pose.txt'
            pose = np.loadtxt(os.path.join(img_id_dir,posepath))
            R = pose[:3,:3]
            angles = rotation_matrix_to_euler_angles(R)
            f.write(os.path.join(person_id,imagepath)+' ')
            for num in angles:
                f.write(str(num)+' ')
            f.write('\n')
        print(test_num)
    mf.close()
    ff.close()
    test_ff.close()
    test_mf.close()


class ImageList(object):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, image_list, labels=None, transform=None, target_transform=None,
                 loader=default_loader):
        imgs = make_dataset_r(image_list, labels)
        if len(imgs) == 0:
            raise FileNotFoundError

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.root = '/home/s4686009/remotedata/BiwiKinect/faces_0'

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        path, target = self.imgs[index]
        path = os.path.join(self.root, path)
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target,index

    def __len__(self):
        return len(self.imgs)

if __name__=='__main__':
    # root = '/home/s4686009/remotedata/BiwiKinect'
    # datas = make_dataset_file(root)

    datalist1 = ImageList(open('male.txt').readlines(), transform=None)
    datalist2 = ImageList(open('male_test.txt').readlines(), transform=None)
    datalist3 = ImageList(open('fmale.txt').readlines(), transform=None)
    datalist4 = ImageList(open('fmale_test.txt').readlines(), transform=None)
    for i,t,_ in datalist1:
        if len(t)==2:
            print('error')
    for i,t,_ in datalist2:
        if len(t)==2:
            print('error')
    for i,t,_ in datalist3:
        if len(t)==2:
            print('error')
    for i,t,_ in datalist4:
        if len(t)==2:
            print('error')