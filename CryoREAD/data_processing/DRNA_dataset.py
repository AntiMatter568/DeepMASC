import os
import numpy as np
import torch
import torch.utils.data
import random
from pathlib import Path
import re

# class Single_Dataset(torch.utils.data.Dataset):
#     def __init__(self, data_path, search_key="input_"):
#         """
#         :param data_path: training data path
#         :param training_id: specify the id that will be used for training
#         """
#         listfiles = [x for x in os.listdir(data_path) if search_key in x]
#         listfiles.sort()
#         self.input_path = []
#         self.id_list = []
#         for x in listfiles:
#             self.input_path.append(os.path.join(data_path, x))
#             cur_id = int(x.replace(search_key, "").replace(".npy", ""))
#             self.id_list.append(cur_id)
#
#     def __len__(self):
#         return len(self.input_path)
#
#     def __getitem__(self, idx):
#         inputfile = self.input_path[idx]
#         cur_id = self.id_list[idx]
#         input = np.load(inputfile)
#         input = input[np.newaxis, :]
#         input = torch.from_numpy(np.array(input, np.float32, copy=True))
#
#         return input, cur_id

class Single_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path, search_key="input_"):
        """
        :param data_path: training data path
        :param search_key: key to filter input files
        """
        self.data_path = Path(data_path)
        # More efficient file listing and sorting
        self.input_paths = sorted(self.data_path.glob(f"{search_key}*.npy"))

        # Extract IDs more efficiently using regex
        pattern = re.compile(rf"{search_key}(\d+)\.npy")
        self.id_list = [int(pattern.match(x.name).group(1)) for x in self.input_paths]

    def __len__(self):
        return len(self.input_paths)

    def _load_file(self, path):
        # Load array and ensure it's writable with correct dtype
        return np.array(np.load(path), dtype=np.float32, copy=True)

    def __getitem__(self, idx):
        input_path = str(self.input_paths[idx])
        cur_id = self.id_list[idx]

        # Load data ensuring it's writable
        input_data = self._load_file(input_path)

        # Convert to tensor ensuring proper type and shape
        input_tensor = torch.from_numpy(input_data).unsqueeze(0).float()

        return input_tensor, cur_id


class Single_Dataset2(torch.utils.data.Dataset):
    def __init__(self, data_path, search_key="input_"):
        """
        :param data_path: training data path
        :param training_id: specify the id that will be used for training
        """
        listfiles = [x for x in os.listdir(data_path) if search_key in x]
        listfiles.sort()
        self.input_path = []
        self.id_list = []
        for x in listfiles:
            self.input_path.append(os.path.join(data_path, x))
            cur_id = int(x.replace(search_key, "").replace(".npy", ""))
            self.id_list.append(cur_id)

    def __len__(self):
        return len(self.input_path)

    def __getitem__(self, idx):
        inputfile = self.input_path[idx]
        cur_id = self.id_list[idx]
        input = np.load(inputfile)
        input = torch.from_numpy(np.array(input, np.float32, copy=True))
        return input, cur_id
