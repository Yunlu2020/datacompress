import argparse
import time

from transformers import BertForPreTraining, BertTokenizer, TrainingArguments, Trainer, AutoModelForMaskedLM
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel

from torch.utils.data import Dataset, DataLoader

import torch
import os

from transformers import set_seed, \
    get_cosine_with_hard_restarts_schedule_with_warmup, BertConfig
from transformers import DataCollatorForLanguageModeling

from utils.logger import get_root_logger
from datasets import load_dataset

tokenizer = None

class C4DatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        # text = dict(['text', 'url', 'timestamp']
        instance = self.dataset.__getitem__(item)
        return instance['text']


class DataCollatorForLanguageModelingWrap(DataCollatorForLanguageModeling):
    def __call__(self, texts):
        input_ids = tokenizer(texts,
                              padding="longest",
                              max_length=512,
                              truncation=True)['input_ids']
        return super(DataCollatorForLanguageModelingWrap, self).__call__(input_ids)


def main(args):
    num_gpus = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    
    set_seed(args.seed + args.local_rank)
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    logger = get_root_logger(os.path.join(args.output_dir, f'{timestamp}.log'))

    global tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModelingWrap(tokenizer=tokenizer, mlm_probability=args.mlm)

    bert_config = BertConfig.from_pretrained('bert-base-uncased')
    model_without_ddp = AutoModelForMaskedLM.from_config(bert_config).cuda()
    # model_without_ddp = BertForPreTraining(bert_config).cuda()
    # self.net_without_ddp = BertForPreTraining.from_pretrained('bert-base-uncased').cuda()


    model = model_without_ddp
    # if args.distributed:
    #     model = DistributedDataParallel(model_without_ddp,
    #                                        device_ids=[torch.cuda.current_device()],
    #                                        output_device=torch.cuda.current_device(),
    #                                        broadcast_buffers=False,
    #                                        find_unused_parameters=False)
    # else:
    #     model = DataParallel(model_without_ddp)

    # c4_subset = load_dataset('allenai/c4', data_files='en/c4-train.0000*-of-01024.json.gz')
    
    # data_files = {"train": [f"en/c4-train.{i:05d}-of-01024.json.gz" for i in range(12)],
    #               "val": [f"en/c4-train.{i:05d}-of-01024.json.gz" for i in range(12, 14)]}
    #
    # c4_train = load_dataset("allenai/c4", data_files=data_files, split="train", cache_dir='./dataset/')
    # c4_val = load_dataset("allenai/c4", data_files=data_files, split="val", cache_dir='./dataset/')

    data_files = {"train": [f"en/c4-train.{i:05d}-of-01024.json.gz" for i in range(14)]}

    c4_train = load_dataset("allenai/c4", data_files=data_files, split="train", cache_dir='./dataset/')
    c4_val = None

    c4_train = C4DatasetWrapper(c4_train)
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="epoch",
        learning_rate=args.lr,
        num_train_epochs=3,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        do_eval=False,
        warmup_ratio=0.1,
        save_strategy='epoch',
        per_device_train_batch_size=args.batch_size,
        # debug=args.debug, # no used
        resume_from_checkpoint=args.resume_from,
        gradient_accumulation_steps=args.gradient_accumulation_steps
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=c4_train,
        eval_dataset=c4_val,
        data_collator=data_collator,
        # no_deprecation_warning=True
    )

    trainer.train()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=0)
    
    parser.add_argument("--c4_root", type=str, default='./dataset/c4/')
    parser.add_argument("--batch_size", type=int, default=16)
    
    parser.add_argument("--mlm", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--betas", type=list, default=(0.9, 0.999))
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=100)
    
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--resume_from", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default='./bert_base_log/')
    args = parser.parse_args()
    
    main(args)
