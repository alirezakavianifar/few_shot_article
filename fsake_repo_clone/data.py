from __future__ import print_function
from torchtools import *
import torch.utils.data as data
import random
import os
import numpy as np
from PIL import Image as pil_image
import pickle
from itertools import islice
from torchvision import transforms
from   tqdm import tqdm
import cv2

class MiniImagenetLoader(data.Dataset):
    def __init__(self, root, partition='train'):
        super(MiniImagenetLoader, self).__init__()
        # set dataset information
        self.root = root
        self.partition = partition
        self.data_size = [3, 84, 84]

        # set normalizer
        mean_pix = [x / 255.0 for x in [120.39586422, 115.59361427, 104.54012653]]
        std_pix = [x / 255.0 for x in [70.68188272, 68.27635443, 72.54505529]]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.RandomCrop(84, padding=4),
                                                 lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])

        # load data
        self.data = self.load_dataset()

    def load_dataset(self):
        # load data
        dataset_path = os.path.join(self.root, 'mini-imagenet', 'compacted_datasets', 'mini_imagenet_%s.pickle' % self.partition)
        try:
            with open(dataset_path, 'rb') as handle:
                data = pickle.load(handle)
        except:
            with open(dataset_path, 'rb') as handle:
                data = pickle.load(handle, encoding='latin1')
        
        print(f"DEBUG: Loaded data type: {type(data)}")
        print(f"DEBUG: Data is dict: {isinstance(data, dict)}")
        
        if isinstance(data, dict):
            # Check if this is the new format (catname2label, labels, data)
            if 'catname2label' in data and 'labels' in data and 'data' in data:
                print("DEBUG: Detected split format (catname2label, labels, data)")
                labels = data['labels']
                images = data['data']
                
                # Reorganize: group images by label
                data_by_class = {}
                for label, image in zip(labels, images):
                    if label not in data_by_class:
                        data_by_class[label] = []
                    data_by_class[label].append(image)
                data = data_by_class
                print(f"DEBUG: Reorganized into {len(data)} classes")
            else:
                # Original format: already organized by class
                print(f"DEBUG: Data keys: {list(data.keys())[:3]}")
                
                # Normalize dictionary keys to integers
                if data:
                    try:
                        first_key = next(iter(data.keys()))
                        print(f"DEBUG: First key type: {type(first_key)}, value: {repr(first_key)}")
                        
                        # Convert all keys to integers
                        data_converted = {}
                        for k, v in data.items():
                            if isinstance(k, bytes):
                                new_k = int(k.decode('utf-8'))
                            elif isinstance(k, str):
                                new_k = int(k)
                            else:
                                new_k = int(k)
                            data_converted[new_k] = v
                        data = data_converted
                    except Exception as e:
                        print(f"DEBUG: Key conversion failed: {e}. Keys: {list(data.keys())[:3]}")
                        pass

        # for each class
        for c_idx in data:
            # for each image
            for i_idx in range(len(data[c_idx])):
                # resize
                image_data = pil_image.fromarray(np.uint8(data[c_idx][i_idx]))
                image_data = image_data.resize((self.data_size[2], self.data_size[1]))
                data[c_idx][i_idx] = image_data
        return data

    def get_task_batch(self,
                       num_tasks=5,
                       num_ways=20,
                       num_shots=1,
                       num_queries=1,
                       seed=None):

        if seed is not None:
            random.seed(seed)

        # init task batch data
        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            query_data.append(data)
            query_label.append(label)

        # get full class list in dataset
        full_class_list = list(self.data.keys())

        # for each task
        for t_idx in range(num_tasks):
            task_class_list = random.sample(full_class_list, num_ways)

            # for each sampled class in task
            for c_idx in range(num_ways):
                # sample data for support and query (num_shots + num_queries)
                class_data_list = random.sample(self.data[task_class_list[c_idx]], num_shots + num_queries)


                # load sample for support set
                for i_idx in range(num_shots):
                    support_data[i_idx + c_idx * num_shots][t_idx] = self.transform(class_data_list[i_idx])
                    support_label[i_idx + c_idx * num_shots][t_idx] = c_idx

                # load sample for query set
                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = self.transform(class_data_list[num_shots + i_idx])
                    query_label[i_idx + c_idx * num_queries][t_idx] = c_idx

        # convert to tensor (num_tasks x (num_ways * (num_supports + num_queries)) x ...)
        support_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in query_label], 1)

        return [support_data, support_label, query_data, query_label]


class TieredImagenetLoader(object):
    def __init__(self, root, partition='train'):
        super(TieredImagenetLoader, self).__init__()
        self.root = root
        self.partition = partition
        self.data_size = [3, 84, 84]
        if self.partition == 'train':
            self.transform = transforms.RandomCrop(84, padding=4)
        self.data = self.load_data_pickle()
    def load_data_pickle(self):

        print("Loading dataset")
        labels_name = '{}/tiered-imagenet/{}_labels.pkl'.format(self.root, self.partition)#dataset/tiered-imagenet/train_labels.pkl
        images_name = '{}/tiered-imagenet/{}_images.npz'.format(self.root, self.partition)#.npz是对应的图像pkl文件解压后的numpy数组。
        print('labels:', labels_name)
        print('images:', images_name)

        # decompress images if npz not exits
        if not os.path.exists(images_name):
            png_pkl = images_name[:-4] + '_png.pkl'
            if os.path.exists(png_pkl):
                decompress(images_name, png_pkl)
            else:
                raise ValueError('path png_pkl not exits')

        if os.path.exists(images_name) and os.path.exists(labels_name):
            try:
                with open(labels_name, 'rb') as f:
                    data = pickle.load(f)
                    label_specific = data["label_specific"]
            except:
                try:
                    with open(labels_name, 'rb') as f:
                        data = pickle.load(f, encoding='latin1')
                    if b"label_specific" in data:
                        label_specific = data[b"label_specific"]
                    else:
                        label_specific = data["label_specific"]
                except:
                    with open(labels_name, 'rb') as f:
                        data = pickle.load(f, encoding='bytes')
                    label_specific = data[b'label_specific']
            print('read label data:{}'.format(len(label_specific)))
        labels = label_specific

        with np.load(images_name, mmap_mode="r", encoding='latin1') as data:
            image_data = data["images"]
            print('read image data:{}'.format(image_data.shape))


        data = {}
        n_classes = np.max(labels) + 1
        for c_idx in range(n_classes):
            data[c_idx] = []
            idxs = np.where(labels==c_idx)[0]
            np.random.RandomState(tt.arg.seed).shuffle(idxs)
            for i in idxs:
                image2resize = pil_image.fromarray(np.uint8(image_data[i,:,:,:]))
                image_resized = image2resize.resize((self.data_size[2], self.data_size[1]))
                if self.partition == 'train':
                    image_resized = self.transform(image_resized)
                image_resized = np.array(image_resized, dtype='float32')

                # Normalize
                image_resized = np.transpose(image_resized, (2, 0, 1))#C,H,W
                image_resized[0, :, :] -= 120.45  # R
                image_resized[1, :, :] -= 115.74  # G
                image_resized[2, :, :] -= 104.65  # B
                image_resized /= 127.5
                data[c_idx].append(image_resized)
        return data


    def get_task_batch(self,
                       num_tasks=5,
                       num_ways=20,
                       num_shots=1,
                       num_queries=1,
                       seed=None):
        if seed is not None:
            random.seed(seed)

        # init task batch data
        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                             dtype='float32')
            query_data.append(data)
            query_label.append(label)

        # get full class list in dataset
        full_class_list = list(self.data.keys())

        # for each task
        for t_idx in range(num_tasks):
            # define task by sampling classes (num_ways)
            task_class_list = random.sample(full_class_list, num_ways)

            # for each sampled class in task
            for c_idx in range(num_ways):
                class_data_list = random.sample(self.data[task_class_list[c_idx]], num_shots + num_queries)

                # load sample for support set
                for i_idx in range(num_shots):
                    # set data
                    support_data[i_idx + c_idx * num_shots][t_idx] = class_data_list[i_idx]
                    support_label[i_idx + c_idx * num_shots][t_idx] = c_idx

                # load sample for query set
                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = class_data_list[num_shots + i_idx]
                    query_label[i_idx + c_idx * num_queries][t_idx] = c_idx

        # convert to tensor (num_tasks x (num_ways * (num_supports + num_queries)) x ...)
        support_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in query_label], 1)

        return [support_data, support_label, query_data, query_label]

class Cub200Loader(data.Dataset):
    def __init__(self, root, partition='train'):
        super(Cub200Loader, self).__init__()
        # set dataset information
        self.root = root
        self.partition = partition
        self.data_size = [3, 84, 84]

        # set normalizer
        mean_pix = [0.485, 0.456, 0.406]
        std_pix = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean=mean_pix, std=std_pix)

        # set transformer
        if self.partition == 'train':
            self.transform = transforms.Compose([transforms.RandomCrop(84, padding=4),
                                                 lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])
        else:  # 'val' or 'test' ,
            self.transform = transforms.Compose([lambda x: np.asarray(x),
                                                 transforms.ToTensor(),
                                                 normalize])

        # load data
        self.data = self.load_dataset()

    def load_dataset(self):
            # load data
            dataset_path = os.path.join(self.root,'cub-200-2011', 'cub200_%s.pickle' % self.partition)
            try:
                with open(dataset_path, 'rb') as handle:
                    data = pickle.load(handle)
            except:
                with open(dataset_path, 'rb') as handle:
                    data = pickle.load(handle, encoding='latin1')
            
            # Check if this is split format
            if isinstance(data, dict) and 'catname2label' in data and 'labels' in data and 'data' in data:
                labels = data['labels']
                images = data['data']
                data_by_class = {}
                for label, image in zip(labels, images):
                    if label not in data_by_class:
                        data_by_class[label] = []
                    data_by_class[label].append(image)
                data = data_by_class
            
            # Normalize keys to integers
            if isinstance(data, dict):
                data_converted = {}
                for k, v in data.items():
                    if isinstance(k, bytes):
                        new_k = int(k.decode('utf-8'))
                    elif isinstance(k, str):
                        new_k = int(k)
                    else:
                        new_k = int(k)
                    data_converted[new_k] = v
                data = data_converted

            # for each class
            for c_idx in data:
                # for each image
                for i_idx in range(len(data[c_idx])):
                    # resize
                    image_data = pil_image.fromarray(np.uint8(data[c_idx][i_idx]),'RGB')
                    image_data = image_data.resize((self.data_size[2], self.data_size[1]))

                    # save
                    data[c_idx][i_idx] = image_data
            return data

    def get_task_batch(self,
                    num_tasks=5,
                    num_ways=20,
                    num_shots=1,
                    num_queries=1,
                    seed=None):

        if seed is not None:
            random.seed(seed)

        # init task batch data
        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                                dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                                dtype='float32')
            query_data.append(data)
            query_label.append(label)

        # get full class list in dataset
        full_class_list = list(self.data.keys())

        # for each task
        for t_idx in range(num_tasks):
            # define task by sampling classes (num_ways)
            task_class_list = random.sample(full_class_list, num_ways)

            # for each sampled class in task
            for c_idx in range(num_ways):
                # sample data for support and query (num_shots + num_queries)
                class_data_list = random.sample(self.data[task_class_list[c_idx]], num_shots + num_queries)


                # load sample for support set
                for i_idx in range(num_shots):
                    # set data
                    support_data[i_idx + c_idx * num_shots][t_idx] = self.transform(class_data_list[i_idx])
                    support_label[i_idx + c_idx * num_shots][t_idx] = c_idx

                # load sample for query set
                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = self.transform(class_data_list[num_shots + i_idx])
                    query_label[i_idx + c_idx * num_queries][t_idx] = c_idx

        # convert to tensor (num_tasks x (num_ways * (num_supports + num_queries)) x ...)
        support_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in query_label], 1)

        return [support_data, support_label, query_data, query_label]

class CifarFsLoader(data.Dataset):
    def __init__(self, root, partition='train'):
        super(CifarFsLoader, self).__init__()
        self.root = root
        self.partition = partition
        self.data_size = [3, 84, 84]



        # set transformer
        if self.partition == 'train':
            image_size = 84
            self.transform = transforms.Compose([
                        transforms.RandomResizedCrop(image_size),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                             np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        else:  # 'val' or 'test' ,
            image_size = 84
            self.transform = transforms.Compose([
                        transforms.Resize([92, 92]),
                        transforms.CenterCrop(image_size),
                        transforms.ToTensor(),
                        transforms.Normalize(np.array([x / 255.0 for x in [125.3, 123.0, 113.9]]),
                                             np.array([x / 255.0 for x in [63.0, 62.1, 66.7]]))])
        # load data
        self.data = self.load_dataset()

    def load_dataset(self):
        # load data
        dataset_path = os.path.join(self.root, 'cifar_fs', '%s.pickle' % self.partition)
        try:
            with open(dataset_path, 'rb') as fo:
                data = pickle.load(fo)
        except:
            with open(dataset_path, 'rb') as f:
                u = pickle._Unpickler(f)
                u.encoding = 'latin1'
                data = u.load()
        
        # Check if this is split format
        if isinstance(data, dict) and 'catname2label' in data and 'labels' in data and 'data' in data:
            labels = data['labels']
            images = data['data']
            data_by_class = {}
            for label, image in zip(labels, images):
                if label not in data_by_class:
                    data_by_class[label] = []
                data_by_class[label].append(image)
            data = data_by_class
        
        # Normalize keys to integers
        if isinstance(data, dict):
            data_converted = {}
            for k, v in data.items():
                if isinstance(k, bytes):
                    new_k = int(k.decode('utf-8'))
                elif isinstance(k, str):
                    new_k = int(k)
                else:
                    new_k = int(k)
                data_converted[new_k] = v
            data = data_converted
            
        data_c = {}

        for c_idx in data:
            # for each image
            for i_idx in range(len(data[c_idx])):
                # resize
                image_data = pil_image.fromarray(np.uint8(data[c_idx][i_idx]))
                image_data = image_data.resize((self.data_size[2], self.data_size[1]))

                # save
                data[c_idx][i_idx] = image_data
        return data

    def get_task_batch(self,
                    num_tasks=5,
                    num_ways=20,
                    num_shots=1,
                    num_queries=1,
                    seed=None):

        if seed is not None:
            random.seed(seed)

        # init task batch data
        support_data, support_label, query_data, query_label = [], [], [], []
        for _ in range(num_ways * num_shots):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                                dtype='float32')
            support_data.append(data)
            support_label.append(label)
        for _ in range(num_ways * num_queries):
            data = np.zeros(shape=[num_tasks] + self.data_size,
                            dtype='float32')
            label = np.zeros(shape=[num_tasks],
                                dtype='float32')
            query_data.append(data)
            query_label.append(label)

        # get full class list in dataset
        full_class_list = list(self.data.keys())

        # for each task
        for t_idx in range(num_tasks):
            # define task by sampling classes (num_ways)
            task_class_list = random.sample(full_class_list, num_ways)

            # for each sampled class in task
            for c_idx in range(num_ways):
                # sample data for support and query (num_shots + num_queries)
                class_data_list = random.sample(self.data[task_class_list[c_idx]], num_shots + num_queries)


                # load sample for support set
                for i_idx in range(num_shots):
                    # set data
                    support_data[i_idx + c_idx * num_shots][t_idx] = self.transform(class_data_list[i_idx])
                    support_label[i_idx + c_idx * num_shots][t_idx] = c_idx

                # load sample for query set
                for i_idx in range(num_queries):
                    query_data[i_idx + c_idx * num_queries][t_idx] = self.transform(class_data_list[num_shots + i_idx])
                    query_label[i_idx + c_idx * num_queries][t_idx] = c_idx

        # convert to tensor (num_tasks x (num_ways * (num_supports + num_queries)) x ...)
        support_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in support_data], 1)
        support_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in support_label], 1)
        query_data = torch.stack([torch.from_numpy(data).float().to(tt.arg.device) for data in query_data], 1)
        query_label = torch.stack([torch.from_numpy(label).float().to(tt.arg.device) for label in query_label], 1)

        return [support_data, support_label, query_data, query_label]


def decompress(path, output):
  with open(output, 'rb') as f:
    array = pickle.load(f, encoding='bytes')
  images = np.zeros([len(array), 84, 84, 3], dtype=np.uint8)
  for ii, item in tqdm(enumerate(array), desc='decompress'):
    im = cv2.imdecode(item, 1)
    images[ii] = im
  np.savez(path, images=images)