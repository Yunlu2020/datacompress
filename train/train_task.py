import torch
import torch.nn as nn
import time
from tqdm import tqdm
from modelutils import BertClassfilyModel, BertForQuestionAnswering
from torch.optim import Adam
from dataset import TaskDataset
from settings import Settings
import os


class Train:
    '''
       该类是三个下游任务的模型训练类，从setting中加载参数，并指定三个下游任务中的某个任务
    '''
    
    def __init__(self, task: str, setting: Settings):
        self.setting = setting
        self.bertbaseweight = setting.bertbaseweight
        self.net = self.build_Model()
        task_list = ['SST-2', 'MNLI', 'SQuAD1.1']
        if task not in task_list:
            raise Exception("Not have this task {},the task you choose must be :SST-2、MNLI、SQuAD1.1".format(task))
        self.task = task
    
    def build_Model(self):
        '''
            给bert模型加载任务头，得到针对三个下游任务的模型
        '''
        if self.task == 'SST':
            net = BertClassfilyModel(class_size=2)
        elif self.task == 'MNLI':
            net = BertClassfilyModel(class_size=3)
        else:
            net = BertForQuestionAnswering()
        return net
    
    def load_bert_model(self):
        '''
            从指定bert模型加载至指定任务模型中去
        '''
        bert_dict = torch.load(self.bertbaseweight).state_dict()
        bert_dict_list = list(bert_dict.keys())
        net_dict = self.net.state_dict()
        net_dict_list = list(net_dict.keys())
        for i in range(len(bert_dict_list)):
            net_dict[net_dict_list[i]] = bert_dict[bert_dict_list[i]]
        self.net.load_state_dict(net_dict)
    
    def load_data(self):
        '''
            得到训练集合、测试集、验证集
        '''
        ret = TaskDataset.Ruturn_loader(task=self.task, settings=self.setting)
        dataloaderdict = ret()
        return dataloaderdict
    
    def save_pretrained(model, path):
        # 保存模型，先利用os模块创建文件夹，后利用torch.save()写入模型文件
        os.makedirs(path, exist_ok=True)
        torch.save(model, os.path.join(path, 'model.pth'))
    
    def train_epoch(self):
        '''
            加载模型，加载数据、然后开始训练，每轮训练结束计算总Loss
        '''
        timestamp = time.strftime("%m_%d_%H_%M", time.localtime())
        optimizer = Adam(self.net.parameters(), self.setting.learning_rate)
        CE_loss = nn.CrossEntropyLoss()
        loader_dict = self.load_data()
        train_dataloader = loader_dict['train']
        val_dataloader = loader_dict['val']
        device = 'cuda'
        self.load_bert_model()
        self.net.train()
        for epoch in range(1, + self.setting.num_epoch + 1):
            total_loss = 0
            for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch}"):
                inputs, targets = [x.to(device) for x in batch]
                optimizer.zero_grad()
                # 模型前向传播，model(inputs)等同于model.forward(inputs)
                loss = self.net(inputs, targets)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                # 测试过程
                # acc统计模型在测试数据上分类结果中的正确个数
            acc = 0
            for batch in tqdm(val_dataloader, desc=f"Testing"):
                inputs, targets = [x.to(device) for x in batch]
                with torch.no_grad():
                    bert_output = self.net(inputs)
                    acc += (bert_output.argmax(dim=1) == targets).sum().item()
            # 输出在测试集上的准确率w
            print(f"Acc: {acc / len(val_dataloader):.2f}")
            
            if epoch % self.setting.check_step == 0:
                # 保存模型
                checkpoints_dirname = "bert_{}".format(self.task) + timestamp
                os.makedirs(checkpoints_dirname, exist_ok=True)
                self.save_pretrained(self.net,
                                     checkpoints_dirname + '/checkpoints-{}/'.format(epoch))
    
    def test(self):
        NotImplemented
