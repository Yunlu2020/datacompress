import argparse
import random
import time

from transformers import BertForPreTraining, BertTokenizer
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from dataset.C4dataset import Uncase_C4, coofn
from torch.utils.data import Dataset, DataLoader

import torch
import os

from transformers import set_seed, \
    get_cosine_with_hard_restarts_schedule_with_warmup, BertConfig
from transformers import DataCollatorForLanguageModeling

from optim_schedule import ScheduledOptim
from utils.checkpoint import save_checkpoint
from utils.logger import get_root_logger


class Train():
    def __init__(self, args):
        self.args = args
        
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        logger = get_root_logger(os.path.join(args.output_dir, f'{timestamp}.log'))
        
        self.tokenizer = BertTokenizer('bert-base-uncased')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=0.15)

        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        self.net_without_ddp = BertForPreTraining(bert_config).cuda()
        # self.net_without_ddp = BertForPreTraining.from_pretrained('bert-base-uncased').cuda()
        
        if args.distributed:
            self.net = DistributedDataParallel(self.net_without_ddp,
                                               device_ids=[torch.cuda.current_device()],
                                               output_device=torch.cuda.current_device(),
                                               broadcast_buffers=False,
                                               find_unused_parameters=False)
        else:
            self.net = DataParallel(self.net_without_ddp)
        
        self.optimizer = torch.optim.Adam(self.net.parameters(),
                                          lr=args.lr,
                                          betas=args.betas,
                                          weight_decay=args.weight_decay)
        self.path = args.c4_root
        self.epoch = 3
        self.rotation = 0.8
        
        self.iter_save = 1
        self.optim_schedule = ScheduledOptim(self.optimizer, 768, n_warmup_steps=args.warmup_steps)
        
        self.loader_data()
        
        logger.info("Create dataloader")
    
    def loadpath(self):
        '''
        载入多个训练和测试文件
        '''
        file_list = os.listdir(self.path)
        train_len = round(len(file_list) * 0.8)
        self.path_list = {'train': [], 'test': []}
        file_list_train = random.sample(file_list, train_len)
        for file in file_list:
            if file not in file_list_train:
                self.path_list['test'].append(os.path.join(self.path, file))
            else:
                self.path_list['train'].append(os.path.join(self.path, file))
    
    def loader_data(self):
        '''
            根据测试文件和训练文件载入dataloader
        '''

        logger = get_root_logger()

        self.loadpath()
        train_datasets = [Uncase_C4(root=path) for path in self.path_list['train'] if path.endswith('.json.gz')]
        test_datasets = [Uncase_C4(root=path) for path in self.path_list['test'] if path.endswith('.json.gz')]

        logger.info(f'Dataset: {sum([len(dataset) for dataset in train_datasets])} instances in train')
        logger.info(f'Dataset: {sum([len(dataset) for dataset in test_datasets])} instances in test')
        
        if self.args.distributed:
            train_samplers = [torch.utils.data.DistributedSampler(dataset, shuffle=True) for dataset in train_datasets]
            test_samplers = [torch.utils.data.DistributedSampler(dataset, shuffle=False) for dataset in test_datasets]
        else:
            train_samplers = [None] * len(train_datasets)
            test_samplers = [None] * len(test_datasets)
            
        self.loaders = {'train':
                            [DataLoader(dataset, batch_size=self.args.batch_size, collate_fn=coofn, sampler=sampler)
                             for dataset, sampler in zip(train_datasets, train_samplers)],
                        'test':
                            [DataLoader(dataset, batch_size=self.args.batch_size, collate_fn=coofn, sampler=sampler)
                             for dataset, sampler in zip(test_datasets, test_samplers)]
                        }
    
    def train_loader(self, epoch, loader):
        '''
            训练每个Loader
        '''
        
        logger = get_root_logger()
        
        total_loss = 0
        self.net.train()
        for i, batch in loader:
            # token_ids, segments, attentionmask, all_mlm_labels, nsp_labels = [x.cuda() for x in batch]
            # token_ids, segments, attentionmask, all_mlm_labels, nsp_labels = [x.cuda() for x in batch]

            self.data_collator(batch['sentences'])
            
            self.optimizer.zero_grad()
            
            loss = self.net(input_ids=token_ids,
                            token_type_ids=segments,
                            attention_mask=attentionmask,
                            labels=all_mlm_labels,
                            next_sentence_label=nsp_labels)
            
            loss.backward()
            # self.optimizer.step()
            self.optim_schedule.step_and_update_lr()
            total_loss += loss.item()
            if i % self.args.log_interval == 0:
                logger.info('average loss for 1000th', total_loss / self.args.log_interval)
                total_loss = 0
        
        save_checkpoint(self.net,
                        os.path.join(self.args.output_dir, f'epoch_{epoch}.pth'),
                        self.optimizer, scaler=None, scheduler=None, meta=dict(epoch=epoch))
    
    def train_net(self):
        '''
            将这个网络进行训练
        '''
        
        for i in range(self.epoch):
            for loader in self.loaders['train']:
                self.train_loader(epoch=i, loader=loader)
    
    def test(self):
        '''
            test
        '''
        logger = get_root_logger()
        count = 0
        total_loss = 0
        for loader in self.loaders['test']:
            for i, batch in enumerate(loader):
                total_loss = 0
                with torch.no_grad():
                    token_ids, segments, attentionmask, all_mlm_labels, nsp_labels = [x.cuda() for x in batch]
                    loss = self.net(input_ids=token_ids, token_type_ids=segments, attention_mask=attentionmask,
                                    masked_lm_labels=all_mlm_labels,
                                    next_sentence_label=nsp_labels)
                    total_loss += loss.item()
                if i % self.args.log_interval:
                    logger.info(f'test total loss: {total_loss}')
                count += 1
        
        logger.info(f'test total loss: {total_loss}, mean loss: {total_loss / count}')


def main(args):
    num_gpus = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    
    set_seed(args.seed + args.local_rank)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    trainer = Train(args)
    trainer.train_net()
    trainer.test()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=0)
    
    parser.add_argument("--c4_root", type=str, default='./dataset/c4/')
    parser.add_argument("--batch_size", type=int, default=128)
    
    parser.add_argument("--mlm", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--betas", type=list, default=(0.9, 0.999))
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--output_dir", type=str, default='./bert_base_log/')
    args = parser.parse_args()
    
    main(args)
