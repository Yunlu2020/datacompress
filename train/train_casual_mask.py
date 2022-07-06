import argparse
import time

from src.utils.dist import is_master
from transformers import BertForPreTraining, BertTokenizer, TrainingArguments, Trainer, AutoModelForMaskedLM, \
    AutoModelForCausalLM
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
from torch.utils.data import Subset

tokenizer = None

class C4DatasetWrapper(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, item):
        # text = dict(['text', 'url', 'timestamp']
        if isinstance(item, torch.Tensor):
            item = item.item()
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
    
    if is_master() and not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    logger = get_root_logger(os.path.join(args.output_dir, f'{timestamp}.log'))

    global tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModelingWrap(tokenizer=tokenizer, mlm=False)

    bert_config = BertConfig.from_pretrained('bert-base-uncased', is_decoder=True)
    model_without_ddp = AutoModelForCausalLM.from_config(bert_config).cuda()

    model = model_without_ddp

    data_files = {"train": [f"en/c4-train.{i:05d}-of-01024.json.gz" for i in range(14)]}

    c4_train = load_dataset("allenai/c4", data_files=data_files, split="train", cache_dir='./dataset/')
    c4_val = None
    
    c4_train = C4DatasetWrapper(c4_train)

    dataset_generator = torch.Generator()
    dataset_generator.manual_seed(42)
    randperm = torch.randperm(len(c4_train), generator=dataset_generator)

    train_len = int(len(c4_train) * args.ratio)

    train_indices = randperm[:train_len]
    val_indices = randperm[train_len:]

    # We then pass the original dataset and the indices we are interested in
    train_subset = Subset(c4_train, train_indices)
    val_subset = Subset(c4_train, val_indices)
    print(c4_train)

    if args.split == 'train':
        train_dataset = train_subset
        logger.info("Using train set")
    elif args.split == 'val':
        train_dataset = val_subset
        logger.info("Using val set")
    else:
        raise RuntimeError("Wrong split. Got {}.".format(args.split))
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="no",
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
        train_dataset=train_dataset,
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
    
    parser.add_argument("--ratio", type=float, default=0.7)
    
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--betas", type=list, default=(0.9, 0.999))
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_steps", type=int, default=10000)
    parser.add_argument("--log_interval", type=int, default=100)
    
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    
    parser.add_argument("--debug", action='store_true')
    parser.add_argument("--resume_from", type=str, default=None)

    parser.add_argument("--output_dir", type=str, default='./bert_base_causal_mask_log/')
    parser.add_argument("--split", type=str, default='train')
    args = parser.parse_args()
    
    main(args)
