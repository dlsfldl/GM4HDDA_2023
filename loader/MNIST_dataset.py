import torch
from torchvision.datasets.mnist import MNIST

class MNIST_dataset(MNIST):
    def __init__(self,
        root,
        digits=[0,1,2],
        download=True,
        split='training'):

        super(MNIST_dataset, self).__init__(
            root,
            download=download,
        )

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError("Dataset not found." + " You can use download=True to download it")

        self.train = True
        data1, targets1 = self._load_data()
        self.train = False
        data2, targets2 = self._load_data()

        data = (torch.cat([data1, data2], dim=0).to(torch.float32) / 255).unsqueeze(1)
        targets = torch.cat([targets1, targets2], dim=0)

        if digits == "all":
            pass
        else:
            data_list = []
            targets_list = []
            for d, t in zip(data, targets):
                if t in digits:
                    data_list.append(d.unsqueeze(0))
                    targets_list.append(t.unsqueeze(0))
            data = torch.cat(data_list, dim=0)
            targets = torch.cat(targets_list, dim=0)

        split_train_val_test = (5/7, 1/7, 1/7)
        num_train_data = int(len(data) * split_train_val_test[0])
        num_valid_data = int(len(data) * split_train_val_test[1]) 

        if split == "training":
            data = data[:num_train_data]
            targets = targets[:num_train_data]
        elif split == "validation":
            data = data[num_train_data:num_train_data + num_valid_data]
            targets = targets[num_train_data:num_train_data + num_valid_data]
        elif split == "test":
            data = data[num_train_data + num_valid_data:]
            targets = targets[num_train_data + num_valid_data:]

        self.data = data
        self.targets = targets

        print(f"MNIST split {split} | {self.data.size()}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y