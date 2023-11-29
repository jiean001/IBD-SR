import os
import numpy as np
import pandas as pd
import pickle
import torch
import scipy.ndimage as ndi
import sklearn.preprocessing as preprocessing
from torch.utils.data import Dataset


def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


def apply_transform(x, transform, fill_mode='nearest', fill_value=0.):
    x = x.astype('float32')
    transform = transform_matrix_offset_center(transform, x.shape[1], x.shape[2])
    final_affine_matrix = transform[:2, :2]
    final_offset = transform[:2, 2]
    channel_images = [
        ndi.interpolation.affine_transform(
            x_channel, final_affine_matrix, final_offset, order=0, mode=fill_mode, cval=fill_value
        ) for x_channel in x
    ]
    x = np.stack(channel_images, axis=0)
    return x


def rotate_image(x3d, theta):
    theta = np.deg2rad(theta)
    rotation_matrix = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1]
        ]
    )
    xrot = apply_transform(x3d, rotation_matrix, 'nearest', 0)
    return xrot


class MyDataset(Dataset):
    def __init__(self, data_tensor, target_tensor, sensitive_tensor):
        super(MyDataset, self).__init__()
        assert data_tensor.shape[0] == target_tensor.shape[0]
        self.data_tensor = torch.from_numpy(np.float32(data_tensor))
        self.target_tensor = torch.from_numpy(np.float32(np.reshape(target_tensor, target_tensor.shape[0])))
        self.sensitive_tensor = torch.from_numpy(np.float32(np.reshape(sensitive_tensor, sensitive_tensor.shape[0])))

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index], self.sensitive_tensor[index]

    def __len__(self):
        return self.data_tensor.shape[0]


class MyDatasetWithoutSensitive(Dataset):
    def __init__(self, data_tensor, target_tensor):
        super(MyDatasetWithoutSensitive, self).__init__()
        assert data_tensor.shape[0] == target_tensor.shape[0]
        self.data_tensor = torch.from_numpy(np.float32(data_tensor))
        self.target_tensor = torch.from_numpy(np.float32(np.reshape(target_tensor, target_tensor.shape[0])))

    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]

    def __len__(self):
        return self.data_tensor.shape[0]


class Adult(Dataset):
    def __init__(self, dataset_path):

        self.dataset_path = dataset_path
        self.train_dataset, self.val_dataset, self.test_dataset = self.load()

    @staticmethod
    def number_encode_features(df):
        result = df.copy()
        encoders = {}
        for column in result.columns:
            if result.dtypes[column] == np.object:
                encoders[column] = preprocessing.LabelEncoder()
                result[column] = encoders[column].fit_transform(result[column].astype(str))
        return result, encoders

    def load(self):
        ADULT_RAW_COL_NAMES = [
            "age", "work-class", "fnlwgt", "education", "education-num", "marital-status", "occupation",
            "relationship", "race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"
        ]

        ADULT_RAW_COL_FACTOR = [1, 3, 5, 6, 7, 8, 13]
        ADULT_VALIDATION_SPLIT = 3

        train_data_set_path = os.path.join(self.dataset_path, "adult.data")
        test_data_set_path = os.path.join(self.dataset_path, "adult.test")

        train_data_set = pd.read_table(train_data_set_path, delimiter=", ", header=None, engine='python',
                                       names=ADULT_RAW_COL_NAMES, na_values="?", keep_default_na=False)
        test_data_set = pd.read_table(test_data_set_path, delimiter=", ", header=None, engine='python',
                                      names=ADULT_RAW_COL_NAMES, na_values="?", keep_default_na=False)

        train_data_set.dropna(inplace=True)
        test_data_set.dropna(inplace=True)

        all_data_set = pd.concat([train_data_set, test_data_set])
        all_data_set = pd.get_dummies(all_data_set, columns=[ADULT_RAW_COL_NAMES[i] for i in ADULT_RAW_COL_FACTOR])

        all_data_set.loc[all_data_set.income == ">50K", "income"] = 1.0
        all_data_set.loc[all_data_set.income == ">50K.", "income"] = 1.0
        all_data_set.loc[all_data_set.income == "<=50K", "income"] = 0.0
        all_data_set.loc[all_data_set.income == "<=50K.", "income"] = 0.0

        all_data_set.loc[all_data_set.sex == "Female", "sex"] = 1.0
        all_data_set.loc[all_data_set.sex == "Male", "sex"] = 0.0

        cutoff = train_data_set.shape[0]
        train_data = all_data_set.iloc[:cutoff, (all_data_set.columns != "income") & (all_data_set.columns != "sex")]
        train_sensitive_labels = all_data_set.iloc[:cutoff, all_data_set.columns == "sex"]
        train_labels = all_data_set.iloc[:cutoff, all_data_set.columns == "income"]

        test_data = all_data_set.iloc[cutoff:, (all_data_set.columns != "income") & (all_data_set.columns != "sex")]
        test_sensitive_labels = all_data_set.iloc[cutoff:, all_data_set.columns == "sex"]
        test_labels = all_data_set.iloc[cutoff:, all_data_set.columns == "income"]

        col_valid_in_train_data = [len(train_data.loc[:, x].unique()) > 1 for x in train_data.columns]
        col_valid_in_test_data = [len(test_data.loc[:, x].unique()) > 1 for x in test_data.columns]

        col_valid = list(map(lambda x, y: x and y, col_valid_in_train_data, col_valid_in_test_data))
        train_data = train_data.loc[:, col_valid]
        test_data = test_data.loc[:, col_valid]

        cutoff = int(np.floor(train_data_set.shape[0] / ADULT_VALIDATION_SPLIT))
        index = np.random.permutation(train_data_set.shape[0])

        val_data = train_data.iloc[index, :].iloc[:cutoff, :]
        val_labels = train_labels.iloc[index, :].iloc[:cutoff, :]
        val_sensitive_labels = train_sensitive_labels.iloc[index, :].iloc[:cutoff, :]

        train_data = train_data.iloc[index, :].iloc[cutoff:, :]
        train_labels = train_labels.iloc[index, :].iloc[cutoff:, :]
        train_sensitive_labels = train_sensitive_labels.iloc[index, :].iloc[cutoff:, :]

        # data normalization
        maxes = np.maximum(np.maximum(train_data.max(axis=0), val_data.max(axis=0)), test_data.max(axis=0))

        train_data = train_data.values / maxes.values
        val_data = val_data.values / maxes.values
        test_data = test_data.values / maxes.values

        train_dataset = MyDataset(train_data, train_labels.values, train_sensitive_labels.values)
        val_dataset = MyDataset(val_data, val_labels.values, val_sensitive_labels.values)
        test_dataset = MyDataset(test_data, test_labels.values, test_sensitive_labels.values)

        return train_dataset, val_dataset, test_dataset


class ExtendedYaleB(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.train_dataset, self.test_dataset = self.load()

    def load(self):
        data1 = pickle.load(open(self.dataset_path + "/set_0.pdata", "rb"), encoding='latin1')
        data2 = pickle.load(open(self.dataset_path + "/set_1.pdata", "rb"), encoding='latin1')
        data3 = pickle.load(open(self.dataset_path + "/set_2.pdata", "rb"), encoding='latin1')
        data4 = pickle.load(open(self.dataset_path + "/set_3.pdata", "rb"), encoding='latin1')
        data5 = pickle.load(open(self.dataset_path + "/set_4.pdata", "rb"), encoding='latin1')
        test = pickle.load(open(self.dataset_path + "/test.pdata", "rb"), encoding='latin1')

        train_data = np.concatenate(
            (data1["x"], data2["x"], data3["x"], data4["x"], data5["x"]), axis=0)
        train_labels = np.concatenate(
            (data1["t"], data2["t"], data3["t"], data4["t"], data5["t"]), axis=0)
        train_sensitive_labels = np.concatenate(
            (data1["light"], data2["light"], data3["light"], data4["light"], data5["light"]), axis=0)

        test_data = test["x"]
        test_label = test["t"]
        test_sensitive_label = test["light"]
        index = test_sensitive_label != 5

        test_labels = test_label[index]
        test_sensitive_labels = test_sensitive_label[index]
        test_data = test_data[index]

        train_dataset = MyDataset(train_data / 255.0, train_labels, train_sensitive_labels)
        test_dataset = MyDataset(test_data / 255.0, test_labels, test_sensitive_labels)

        return train_dataset, test_dataset


class MNIST_ROT(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = os.path.join(dataset_path, 'MNIST-ROT')
        self.train_dataset, self.test_dataset, self.test_55_dataset, self.test_65_dataset = self.load()

    # def load(self):
    #     train_angle_list = [-45.0, -22.5, 0.0, 22.5, 45.0]
    #     test_55_angle_list = [-55.0, 55.0]
    #     test_65_angle_list = [-65.0, 65.0]
    #
    #     angle_class_map = {angle: label for label, angle in enumerate(train_angle_list)}
    #
    #     train_data_set1, train_label_set1 = torch.load(os.path.join(self.dataset_path, 'training.pt'))
    #     test_data_set, test_label_set = torch.load(os.path.join(self.dataset_path, 'test.pt'))
    #     index = np.random.permutation(range(train_data_set1.shape[0]))
    #     train_index = index[:50000]
    #     valid_index = index[50000:]
    #
    #     train_data_set1 = train_data_set1 / 255.0
    #     test_data_set = test_data_set / 255.0
    #
    #     train_data_set = train_data_set1[train_index]
    #     train_label_set = train_label_set1[train_index]
    #
    #     # valid_data_set = train_data_set1[valid_index]
    #     # valid_label_set = train_label_set1[valid_index]
    #     # concate_data_set = torch.cat((train_data_set, test_data_set), dim=0)
    #     # concate_label_set = torch.cat((train_label_set, test_label_set), dim=0)
    #
    #     train_dataset = MyDatasetWithRotate(train_data_set, train_label_set, train_angle_list, angle_class_map, True)
    #     test_dataset = MyDatasetWithRotate(test_data_set, test_label_set, train_angle_list, angle_class_map, False)
    #     test_55_dataset = MyDatasetWithoutSensitive(test_data_set, test_label_set, test_55_angle_list, False)
    #     test_65_dataset = MyDatasetWithoutSensitive(test_data_set, test_label_set, test_65_angle_list, False)
    #
    #     return train_dataset, test_dataset, test_55_dataset, test_65_dataset

    def load(self):
        train_data = np.load(os.path.join(self.dataset_path, 'train_data.npy')) / 255.0
        train_labels = np.load(os.path.join(self.dataset_path, 'train_labels.npy'))
        train_sensitive_labels = np.load(os.path.join(self.dataset_path, 'train_sensitive_labels.npy'))
        test_data = np.load(os.path.join(self.dataset_path, 'test_data.npy')) / 255.0
        test_labels = np.load(os.path.join(self.dataset_path, 'test_labels.npy'))
        test_sensitive_labels = np.load(os.path.join(self.dataset_path, 'test_sensitive_labels.npy'))

        test_55_data = np.load(os.path.join(self.dataset_path, 'test_55_data.npy')) / 255.0
        test_55_labels = np.load(os.path.join(self.dataset_path, 'test_55_labels.npy'))

        test_65_data = np.load(os.path.join(self.dataset_path, 'test_65_data.npy')) / 255.0
        test_65_labels = np.load(os.path.join(self.dataset_path, 'test_65_labels.npy'))

        train_dataset = MyDataset(train_data, train_labels, train_sensitive_labels)
        test_dataset = MyDataset(test_data, test_labels, test_sensitive_labels)
        test_55_dataset = MyDatasetWithoutSensitive(test_55_data, test_55_labels)
        test_65_dataset = MyDatasetWithoutSensitive(test_65_data, test_65_labels)

        return train_dataset, test_dataset, test_55_dataset, test_65_dataset

    # def load(self):
    #     train_angle_list = [-45.0, -22.5, 0.0, 22.5, 45.0]
    #     test_55_angle_list = [-55.0, 55.0]
    #     test_65_angle_list = [-65.0, 65.0]
    #     angle_class_map = {angle: label for label, angle in enumerate(train_angle_list)}
    #
    #     train_data_set1, train_label_set1 = torch.load(os.path.join(self.dataset_path, 'training.pt'))
    #     # test_data_set, test_label_set = torch.load(os.path.join(self.dataset_path, 'test.pt'))
    #     index = np.random.permutation(range(train_data_set1.shape[0]))
    #     train_index = index[:50000]
    #     valid_index = index[50000:]
    #
    #     train_data_set = train_data_set1[train_index]
    #     train_label_set = train_label_set1[train_index]
    #
    #     test_data_set = train_data_set1[valid_index]
    #     test_label_set = train_label_set1[valid_index]
    #
    #     rot_train_data_set_list = []
    #     rot_train_label_set_list = []
    #     rot_train_sensitive_set_list = []
    #
    #     rot_test_data_set_list = []
    #     rot_test_label_set_list = []
    #     rot_test_sensitive_set_list = []
    #
    #     for index in range(len(train_angle_list)):
    #         angle = train_angle_list[index]
    #         sensitive_label = angle_class_map[angle]
    #         if angle == 0.0:
    #             rot_train_data_set_list.extend(train_data_set.detach().numpy())
    #             rot_test_data_set_list.extend(test_data_set.detach().numpy())
    #         else:
    #             rot_train_data_set = rotate_image(train_data_set.detach().numpy(), angle)
    #             rot_train_data_set_list.extend(rot_train_data_set)
    #
    #             rot_test_data_set = rotate_image(test_data_set.detach().numpy(), angle)
    #             rot_test_data_set_list.extend(rot_test_data_set)
    #
    #         rot_train_label_set_list.extend(train_label_set.detach().numpy())
    #         train_sensitive_set = sensitive_label * torch.ones_like(train_label_set)
    #         rot_train_sensitive_set_list.extend(train_sensitive_set.detach().numpy())
    #
    #         rot_test_label_set_list.extend(test_label_set.detach().numpy())
    #         test_sensitive_set = sensitive_label * torch.ones_like(test_label_set)
    #         rot_test_sensitive_set_list.extend(test_sensitive_set.detach().numpy())
    #
    #     np.save('Data/mnist/MNIST-ROT/train_data.npy', np.asarray(rot_train_data_set_list))
    #     np.save('Data/mnist/MNIST-ROT/train_labels.npy', np.asarray(rot_train_label_set_list))
    #     np.save('Data/mnist/MNIST-ROT/train_sensitive_labels.npy',  np.asarray(rot_train_sensitive_set_list))
    #
    #     np.save('Data/mnist/MNIST-ROT/valid_data.npy', np.asarray(rot_test_data_set_list))
    #     np.save('Data/mnist/MNIST-ROT/valid_labels.npy', np.asarray(rot_test_label_set_list))
    #     np.save('Data/mnist/MNIST-ROT/valid_sensitive_labels.npy',  np.asarray(rot_test_sensitive_set_list))
    #
    #     # print("done!!!!!!")
    #     # rot_test_data_set_list = []
    #     # rot_test_label_set_list = []
    #     #
    #     # for index in range(len(test_65_angle_list)):
    #     #     angle = test_65_angle_list[index]
    #     #     if angle == 0.0:
    #     #         rot_test_data_set_list.extend(test_data_set.detach().numpy())
    #     #     else:
    #     #         rot_test_data_set = self.rotate_image(test_data_set.detach().numpy(), angle)
    #     #         rot_test_data_set_list.extend(rot_test_data_set)
    #     #
    #     #     rot_test_label_set_list.extend(test_label_set.detach().numpy())
    #     #
    #     # np.save('Data/mnist/MNIST-ROT/test_65_data.npy', np.asarray(rot_test_data_set_list))
    #     # np.save('Data/mnist/MNIST-ROT/test_65_labels.npy', np.asarray(rot_test_label_set_list))
    #
    #     return 0


class MNIST_DIL(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = os.path.join(dataset_path, 'MNIST-DIL')
        (self.test_erode_2_dataset, self.test_dilate_2_dataset,
         self.test_dilate_3_dataset, self.test_dilate_4_dataset) = self.load()

    def load(self):

        test_erode_2_data = np.load(os.path.join(self.dataset_path, 'mnist-erode-2-image.npy')) / 255.0
        test_erode_2_labels = np.load(os.path.join(self.dataset_path, 'mnist-erode-2-label.npy'))

        test_dilate_2_data = np.load(os.path.join(self.dataset_path, 'mnist-dilate-2-image.npy')) / 255.0
        test_dilate_2_labels = np.load(os.path.join(self.dataset_path, 'mnist-dilate-2-label.npy'))

        test_dilate_3_data = np.load(os.path.join(self.dataset_path, 'mnist-dilate-3-image.npy')) / 255.0
        test_dilate_3_labels = np.load(os.path.join(self.dataset_path, 'mnist-dilate-3-label.npy'))

        test_dilate_4_data = np.load(os.path.join(self.dataset_path, 'mnist-dilate-4-image.npy')) / 255.0
        test_dilate_4_labels = np.load(os.path.join(self.dataset_path, 'mnist-dilate-4-label.npy'))

        test_erode_2_dataset = MyDatasetWithoutSensitive(test_erode_2_data, test_erode_2_labels)
        test_dilate_2_dataset = MyDatasetWithoutSensitive(test_dilate_2_data, test_dilate_2_labels)
        test_dilate_3_dataset = MyDatasetWithoutSensitive(test_dilate_3_data, test_dilate_3_labels)
        test_dilate_4_dataset = MyDatasetWithoutSensitive(test_dilate_4_data, test_dilate_4_labels)

        return test_erode_2_dataset, test_dilate_2_dataset, test_dilate_3_dataset, test_dilate_4_dataset


class MNIST_ROT_VIS(Dataset):
    def __init__(self, dataset_path):
        self.dataset_path = os.path.join(dataset_path, 'MNIST-ROT')
        self.test_dataset = self.load()

    def load(self):
        num_sample = 100

        test_data = np.load(os.path.join(self.dataset_path, 'test_data.npy')) / 255.0
        test_labels = np.load(os.path.join(self.dataset_path, 'test_labels.npy'))
        test_sensitive_labels = np.load(os.path.join(self.dataset_path, 'test_sensitive_labels.npy'))

        test_data_vis = []
        test_labels_vis = []
        test_sensitive_labels_vis = []

        for class_index in range(10):
            for sensitive_index in range(5):
                num_index = 0
                for data_index in range(len(test_labels)):
                    label = test_labels[data_index]
                    sensitive_label = test_sensitive_labels[data_index]
                    if label == class_index and sensitive_label == sensitive_index:
                        test_data_vis.append(test_data[data_index])
                        test_labels_vis.append(test_labels[data_index])
                        test_sensitive_labels_vis.append(sensitive_label)
                        num_index = num_index + 1

                    if num_index >= num_sample:
                        break

        test_data_vis = np.asarray(test_data_vis)
        test_labels_vis = np.asarray(test_labels_vis)
        test_sensitive_labels_vis = np.asarray(test_sensitive_labels_vis)
        test_dataset = MyDataset(test_data_vis, test_labels_vis, test_sensitive_labels_vis)

        return test_dataset