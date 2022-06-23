from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
from settings import Settings
import torch
import pandas as pd

class BaseDataSet(Dataset):
    '''
        三个任务基本模型
    '''
    def __init__(self,name,data_path):
        self.name = name
        self.data_path = data_path
    def sepecialize(self):
        '''
            通过此函数返回具体任务数据集的子类
        '''
        if self.name == 'SST':
            return SSTDataset(self.data_path)
        if self.name == 'MNLI':
            return MNLIDataset(self.data_path)
        if self.name == 'QA':
            return QADataset(self.data_path)
    def __len__(self):
        return NotImplemented
    def __getitem__(self, item):
        return NotImplemented




class SSTDataset(BaseDataSet):
    '''
        SST任务的数据集格式
    '''
    def __init__(self,data_path):
        self.data = []
        categories = set()
        with open(data_path, 'r', encoding="utf8") as file:
            for sample in file.readlines():
                sent, polar = sample.strip().split("\t")
                categories.add(polar)
                self.data.append((sent, polar))
        self.data = self.data[1:]
        self.data_size = len(self.data)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    def __len__(self):
        return self.data_size
    def __getitem__(self, index):
        return self.data[index]


class MNLIDataset(BaseDataSet):
    def __init__(self, data_path):
        tsv_read = pd.read_csv(data_path, sep='\t')
        self.sentence1 = list(tsv_read['sentence1'])
        self.sentence2 = list(tsv_read['sentence2'])
        self.label = list(tsv_read['gold_label'])
    def __len__(self):
        return len(self.sentence1)
    def __getitem__(self, item):
        return [self.sentence1[item],self.sentence2[item]],self.label[item]

class QADataset(BaseDataSet):
    '''
        QA模型
    '''
    def __init__(self,datapath):
        NotImplemented

def coffate_fn_st2(examples):
    '''
        dataloader输出处理函数
    '''
    inputs, targets = [], []
    for sent, polar in examples:
        inputs.append(sent)
        targets.append(int(polar))
    inputs = tokenizer(inputs,
                       padding=True,
                       truncation=True,
                       return_tensors="pt",
                       max_length=512)
    targets = torch.tensor(targets)
    return inputs, targets

class Ruturn_loader:
    '''
        根据settings 和任务，返回数据
    '''
    def __init__(self, task: str, settings: Settings):
        self.task = task
        self.train_path, self.test_path, self.val_path = settings.task(task).path
        self.batch_size = settings.task(task).bath_size
    def createDataset(self):
        self.train_dataset = BaseDataSet(name=self.task, data_path=self.train_path).sepecialize()
        self.test_dataset = BaseDataSet(name=self.task, data_path=self.test_path).sepecialize()
        self.val_dataset = BaseDataSet(name=self.task, data_path=self.val_path).sepecialize()
    def __call__(self, *args, **kwargs):
        dataloader = {}
        dataloader['train'] = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            collate_fn=coffate_fn_st2,
            shuffle=True
        )
        dataloader['test'] = DataLoader(
            self.test_dataset,
            batch_size=1,
            collate_fn=coffate_fn_st2,
        )
        dataloader['val'] = DataLoader(
            self.val_dataset,
            batch_size=1,
            collate_fn=coffate_fn_st2
        )
        return dataloader




