import argparse
import time

import json

from src.utils.dist import is_master
from transformers import BertForPreTraining, BertTokenizer, \
    TrainingArguments, Trainer, AutoModelForMaskedLM, \
    AutoModelForCausalLM

from model.modeling_bert import BertLMHeadModel
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

from utils.util import fetch_data_by_disk

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
        return dict(text=instance['text'], index=item)


class DataCollatorForLanguageModelingWrap(DataCollatorForLanguageModeling):
    def __call__(self, data):
        input_ids = self.tokenizer(data['text'],
                              padding="longest",
                              max_length=512,
                              truncation=True)['input_ids']
        
        return super(DataCollatorForLanguageModelingWrap, self).__call__(input_ids)


@torch.no_grad()
@torch.cuda.amp.autocast()
def run_inference(args, model, data_loader, data_collator):
    model.eval()
    index2loss = dict()
    
    logger = get_root_logger()

    logger.info(f"Need to run {len(data_loader)} iterations.")
    start_time = time.time()
    for idx, batch in enumerate(data_loader):
        input_ids = {k: v.cuda() for k, v in data_collator(batch).items()}
        output = model(**input_ids)
        for index, loss in zip(batch['index'], output.per_sample_loss):
            index2loss[index.item()] = loss.item()
        
        if (idx + 1) % 100 == 0:
            logger.info(f"Running {idx}/{len(data_loader)}. "
                        f"Per iter time: {(time.time() - start_time)/100}.")
            start_time = time.time()
        
    index2loss = fetch_data_by_disk(index2loss,
                                    os.path.join(args.output_dir,
                                                 f'{args.checkpoint}_{args.eval_split}_loss.pkl'))

    assert len(index2loss) == len(data_loader.dataset)


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
    data_collator = DataCollatorForLanguageModelingWrap(tokenizer=tokenizer, mlm=False)
    
    # bert_config = BertConfig.from_pretrained('bert-base-uncased', is_decoder=True)
    # model_without_ddp = BertLMHeadModel(bert_config).cuda()
    model_without_ddp = BertLMHeadModel.from_pretrained(os.path.join(args.output_dir, args.checkpoint)).cuda()
    
    model = model_without_ddp

    # checkpoint = torch.load(args.pertrained)
    # model.load_state_dict()
    
    data_files = {"train": [f"en/c4-train.{i:05d}-of-01024.json.gz" for i in range(14)]}
    
    c4_dataset = load_dataset("allenai/c4", data_files=data_files, split="train", cache_dir='./dataset/')
    c4_dataset = C4DatasetWrapper(c4_dataset)
    
    logger.info(f"Using {args.eval_split}")
    split_file_path = os.path.join(args.dataset_split_path, f'{args.eval_split}_split_ids.json')
    assert os.path.exists(split_file_path)
    
    dataset_indices = json.load(open(split_file_path, 'r'))

    inference_dataset = Subset(c4_dataset, dataset_indices)

    sampler = None
    if args.distributed:
        sampler = torch.utils.data.DistributedSampler(inference_dataset, shuffle=False)
    data_loader = torch.utils.data.DataLoader(inference_dataset, sampler=sampler,
                                              batch_size=args.batch_size)
    
    run_inference(args, model, data_loader, data_collator)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=0)
    
    parser.add_argument("--c4_root", type=str, default='./dataset/c4/')
    parser.add_argument("--batch_size", type=int, default=64)
    
    parser.add_argument("--ratio", type=float, default=0.7)
    
    parser.add_argument("--checkpoint", type=str, default=None)
    
    parser.add_argument("--dataset_split_path", type=str, default='./dataset_split/round1/')
    parser.add_argument("--eval_split", type=str, default='train')
    
    parser.add_argument("--output_dir", type=str, default='./bert_base_causal_mask_70train_log/')
    args = parser.parse_args()
    
    main(args)
