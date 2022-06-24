import argparse
import os
import time
from collections import defaultdict
from os.path import exists

import json
import mmcv
import pickle
import torch
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DataParallel, DistributedDataParallel
from torch.optim import Adam, Adamax, AdamW
from torch.utils.data import DataLoader

import numpy as np

from transformers import AutoTokenizer, get_linear_schedule_with_warmup, set_seed, \
    get_cosine_with_hard_restarts_schedule_with_warmup

from src.data import create_loader
from src.data.data import PMRDataset
from src.data.utils import build_transform, save_jsonl
from src.model import build_model
from src.model.uniter import UniterForMultimodalReasoning
from src.utils.args import register_tokenizer
from src.utils.checkpoint import save_checkpoint, resume
from src.utils.dist import is_local_master, is_master, all_gather, get_rank, get_world_size, synchronize, broadcast
from src.utils.logger import get_root_logger

assert mmcv.__version__ == '1.3.1'


def build_optimizer(model, args, num_training_steps, warmup_step=-1):
    """ vqa linear may get larger learning rate """
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    param_optimizer = [(n, p) for n, p in model.named_parameters()
                       if 'classifier' not in n]
    param_top = [(n, p) for n, p in model.named_parameters()
                 if 'classifier' in n]
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_top
                    if not any(nd in n for nd in no_decay)],
         'lr': args.lr,
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_top
                    if any(nd in n for nd in no_decay)],
         'lr': args.lr,
         'weight_decay': 0.0},
        {'params': [p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    
    if args.type == 'adam':
        OptimCls = Adam
    elif args.type == 'adamax':
        OptimCls = Adamax
    elif args.type == 'adamw':
        OptimCls = AdamW
    else:
        raise ValueError('invalid optimizer')
    optimizer = OptimCls(optimizer_grouped_parameters,
                         lr=args.lr, betas=args.betas)
    
    warmup_step = warmup_step if warmup_step >= 0 else int(0.1 * num_training_steps)
    
    scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step,
                                                                   num_training_steps=num_training_steps)
    
    return optimizer, scheduler


def train(args, cfg, train_dataloader, val_dataloader, test_dataloader, tokenizer):
    logger = get_root_logger()
    
    model = build_model(cfg.model)
    
    model_without_ddp = model.cuda()
    if args.distributed:
        model = DistributedDataParallel(model_without_ddp,
                                        device_ids=[torch.cuda.current_device()],
                                        output_device=torch.cuda.current_device(),
                                        broadcast_buffers=False,
                                        find_unused_parameters=cfg.get('find_unused_parameters', False))
    else:
        model = DataParallel(model_without_ddp)
    
    gradient_accumulation_steps = cfg.train_tech.gradient_accumulation_steps
    use_amp = cfg.train_tech.use_amp
    t_total = int(len(train_dataloader) * cfg.total_epoch)
    optimizer, scheduler = build_optimizer(model, cfg.optimizer, t_total)
    
    scaler = GradScaler(enabled=cfg.train_tech.use_amp)
    
    count = 0
    count_nan = 0
    cfg.start_epoch = 0
    best_val_acc = -1

    if cfg.resume_from is not None:
        # TODO(HUI): resume
        resume(cfg, model, optimizer, scaler, scheduler)
    
    for epoch in range(cfg.start_epoch, cfg.total_epoch):
        model.train()
        optimizer.zero_grad()
        log_dict = defaultdict(list)
        for iter, batch in enumerate(train_dataloader, 1):
            with torch.cuda.amp.autocast(enabled=use_amp):
                output = model(batch, compute_loss=True)
            loss = output['loss']
            loss = loss / gradient_accumulation_steps
            
            if loss.isnan():
                count_nan += 1
                # if count_nan == 10:
                #     raise RuntimeError("Wrong code. Reach maximum nan loss.")
            else:
                count_nan = 0
            
            scaler.scale(loss).backward()
            
            if (count + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                if cfg.grad_norm == -1:
                    params = list(filter(lambda p: p.requires_grad and p.grad is not None, model.parameters()))
                    grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
            
            for k, v in output.items():
                log_dict[k].append(0 if v.isnan() else v.item())
            
            if (iter + 1) % cfg.log_interval == 0:
                log_string = f'Epoch: {epoch}/{cfg.total_epoch}, iteration: {iter + 1}/{len(train_dataloader)}, grad scaler: {scaler.get_scale():.2f}, '
                log_string += ", ".join([f"{k}: {np.mean(v):.3f}" for k, v in log_dict.items()])
                logger.info(log_string)
                log_dict = defaultdict(list)
            
            count += 1
        
        log_string = f'Epoch: {epoch}, iteration: {iter}, grad scaler: {scaler.get_scale():.2f}, '
        log_string += ", ".join([f"{k}: {np.mean(v):.3f}" for k, v in log_dict.items()])
        logger.info(log_string)
        
        # val_acc, val_loose_acc, total_4cls_acc
        eval_output = evaluate(args=cfg, model=model, dataloader=val_dataloader, epoch=epoch, split='val')
        
        save_checkpoint(model=model_without_ddp,
                        filepath=os.path.join(cfg.output_dir, f'epoch_{epoch}.pth'),
                        optimizer=optimizer,
                        scaler=scaler,
                        scheduler=scheduler,
                        meta=dict(epoch=epoch),
                        )
        log_string = f'[Model Info] Epoch {epoch}/{cfg.total_epoch}: '
        log_string += ", ".join([f"{k}: {v:.4f}" for k, v in eval_output.items()])
        
        logger.info(log_string)
        
        if eval_output['val_acc'] > best_val_acc:
            logger.info(
                f'[Model Info] Saving the best model with best valid val acc {eval_output["val_acc"]:.6f} at epoch {epoch}')
            best_val_acc = eval_output['val_acc']
            
            if is_local_master():
                os.system(f'rm {os.path.join(cfg.output_dir, "best_mode.pth")}')
                mmcv.symlink(os.path.join(cfg.output_dir, f'epoch_{epoch}.pth'),
                             os.path.join(cfg.output_dir, f'best_mode.pth'))
    
    logger.info(f"[Model Info] Best validation performance: {best_val_acc}")
    
    # state_dict = torch.load(os.readlink(os.path.join(cfg.output_dir, 'best_model.pth')))['state_dict']
    # model = UniterForMultimodalReasoning.from_pretrained(cfg.json_config_file, state_dict, img_dim=cfg.image_dim)
    return model


@torch.no_grad()
def evaluate(*, args, model, dataloader, epoch=-1, split='val'):
    logger = get_root_logger()
    
    model.eval()
    logger.info('Start Running Validation')
    
    result_file = os.path.join(args.output_dir, f'epoch{epoch}_{split}.pkl')
    tmp_result_file = os.path.join(args.output_dir, f'epoch{epoch}_rank{get_rank()}_{split}.pkl')
    total_id2score = defaultdict(dict)
    
    num_correct = 0
    num_match_correct = 0
    count = 0
    for iter, batch in enumerate(dataloader):
        total_ids = batch['total_ids']
        img_ids = batch['img_ids']
        ans_pos_idxs = batch['ans_pos_idxs']
        answer_labels = batch['answer_labels']
        
        scores, match_scores = model(batch, compute_loss=False)
        targets = batch['targets'].cuda(non_blocking=True)
        is_answers = batch['is_answers'].cuda(non_blocking=True)
        num_correct += compute_score_with_logits(scores, targets)
        num_match_correct += compute_score_with_logits(match_scores, is_answers)
        count += batch['targets'].shape[0]
        
        for total_id, img_id, score, ans_pos_idx, answer_label in zip(total_ids, img_ids, scores.cpu().numpy(),
                                                                      ans_pos_idxs.numpy(), answer_labels.numpy()):
            if total_id not in total_id2score:
                total_id2score[total_id] = dict(total_id=total_id, img_id=img_id, answer_label=answer_label)
            total_id2score[total_id][ans_pos_idx] = score[0]  # first is the action-true logit.
    
    eval_data = all_gather(torch.LongTensor([num_correct, num_match_correct, count]))
    eval_data = torch.stack(eval_data, dim=0)
    
    total_acc = eval_data[:, 0].sum() / eval_data[:, 2].sum()
    total_loose_acc = eval_data[:, 1].sum() / eval_data[:, 2].sum()
    
    # choose one from four answers
    if is_master():
        synchronize()
        for r in range(1, get_world_size()):
            tmp_result_file = os.path.join(args.output_dir, f'epoch{epoch}_rank{r}_{split}.pkl')
            data = pickle.load(open(tmp_result_file, 'rb'))
            
            for k, v in data.items():
                if k in total_id2score:
                    total_id2score[k].update(v)
                else:
                    total_id2score[k] = v
        
        num_correct = 0
        for k, v in total_id2score.items():
            num_correct += np.argmax([v[i] for i in range(len(v)-3)]) == v['answer_label']
        
        total_4cls_acc = num_correct / len(total_id2score)
        pickle.dump(total_id2score, open(result_file, 'wb'))
        synchronize()
        total_4cls_acc = broadcast(total_4cls_acc)
    else:
        pickle.dump(total_id2score, open(tmp_result_file, 'wb'))
        time.sleep(2)
        synchronize()
        synchronize()
        if os.path.exists(tmp_result_file):
            os.system(f'rm {tmp_result_file}')
        total_4cls_acc = broadcast(None)
    
    model.train()
    return dict(val_acc=total_acc, val_loose_acc=total_loose_acc, val_4cls_acc=total_4cls_acc)


def compute_score_with_logits(logits, labels):
    num_correct = logits.argmax(dim=-1).eq(labels).sum().item()
    return num_correct


def main(args):
    cfg = mmcv.Config.fromfile(args.config)
    
    num_gpus = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = num_gpus > 1
    if args.distributed:
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(args.local_rank)
    
    set_seed(args.seed + args.local_rank)
    
    if is_master() and not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir)
        
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    logger = get_root_logger(os.path.join(cfg.output_dir, f'{timestamp}.log'))
    
    for k in args.__dict__:
        logger.info(f'{k} = {args.__dict__[k]}')
    
    for k in cfg._cfg_dict:
        logger.info(f'{k} = {cfg._cfg_dict[k]}')
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer)
    register_tokenizer(tokenizer)
    
    logger.info('[Data Info] Loading data')
    train_dataloader = create_loader(cfg,
                                     dataset_config=cfg.data.train,
                                     dataloader_config=dict(batch_size=cfg.data.batch_size,
                                                            pin_memory=True,
                                                            num_workers=cfg.data.num_workers))
    
    val_dataloader = create_loader(cfg,
                                   dataset_config=cfg.data.val,
                                   dataloader_config=dict(batch_size=cfg.data.batch_size,
                                                          pin_memory=True,
                                                          drop_last=False,
                                                          num_workers=cfg.data.num_workers))
    
    test_dataloader = None
    if hasattr(cfg.data, 'test'):
        test_dataloader = create_loader(cfg,
                                        dataset_config=cfg.data.test,
                                        dataloader_config=dict(batch_size=cfg.data.batch_size,
                                                               pin_memory=True,
                                                               drop_last=False,
                                                               num_workers=cfg.data.num_workers)
                                        )
    

    
    # training
    logger.info('[Model Info] Start Training')
    model = train(args=args,
                  cfg=cfg,
                  train_dataloader=train_dataloader,
                  val_dataloader=val_dataloader,
                  test_dataloader=test_dataloader,
                  tokenizer=tokenizer)
    # evaluate(args=args, model=model, dataloader=val_dataloader, epoch='best', split='test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=0)
    
    args = parser.parse_args()
    
    main(args)
