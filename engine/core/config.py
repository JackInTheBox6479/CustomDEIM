import torch
from torch.optim import Optimizer
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader


class BaseConfig(object):

    TESTING = False

    def __init__(self):
        super().__init__()

        # instance/function
        self.model = None
        self.postprocessor = None
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.scaler = None
        self.train_dataset = None
        self.val_dataset = None

        #data
        self.num_workers = None
        self.batch_size = None
        self.train_shuffle = None
        self.val_shuffle = None

        #runtime
        self.resume = None
        self.tuning = None
        self.epochs = None
        self.last_epoch = -1
        self.seed = None
        self.device = 'cuda'

    def model(self):
        return self.model

    def model(self, m):
        self.model = m

    def postprocessor(self):
        return self.postprocessor

    def postprocessor(self, m):
        self.postprocessor = m

    def criterion(self):
        return self.criterion

    def criterion(self, m):
        self.criterion = m

    def optimizer(self):
        return self.optimizer

    def optimizer(self, m):
        self.optimizer = m

    def scheduler(self):
        return self.scheduler

    def scheduler(self, m):
        self.scheduler = m

    def train_loader(self):
        if self.train_loader is None and self.train_dataset is not None:
            loader = DataLoader(dataset=self.train_dataset, batch_size=self.batch_size, shuffle=self.train_shuffle, num_workers=self.num_workers)
            loader.shuffle = self.train_shuffle
            self.train_loader = loader
        return self.train_loader

    def train_loader(self, m):
        self.train_loader = m

    def val_loader(self):
        if self.val_loader is None and self.val_dataset is not None:
            loader = DataLoader(dataset=self.val_dataset, batch_size=self.batch_size, shuffle=self.val_shuffle, num_workers=self.num_workers)
            loader.shuffle = self.val_shuffle
            self.val_loader = loader
        return self.val_loader

    def scaler(self):
        if self.scaler is None and torch.cuda.is_available():
            self.scaler = GradScaler()
        return self.scaler

    def scaler(self, m):
        self.scaler = m

    def val_shuffle(self):
        if self.val_shuffle is None:
            return False
        return self.val_shuffle

    def val_shuffle(self, m):
          self.val_shuffle = m

    def train_shuffle(self):
        if self.train_shuffle is None:
            return True
        return self.train_shuffle

    def train_shuffle(self, m):
        self.train_shuffle = m

    def batch_size(self):
        if self.batch_size is None:
            return None
        return self.batch_size

    def batch_size(self, m):
        self.batch_size = m

    def train_dataset(self):
        return self.train_dataset

    def train_dataset(self, dataset):
        self.train_dataset = dataset

    def val_dataset(self):
        return self.val_dataset

    def val_dataset(self, dataset):
        self.val_dataset = dataset
