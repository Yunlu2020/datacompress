# datacompress
预训练中的数据选取

dataset文件夹下包含三个下游任务数据集文件
train文件夹下包含三个下游任务和bert预训练任务代码
modelutils包含三个模型分别对应三个任务
setting文件为配置文件

# train
pre-train bert on 4 3090 GPU. Run the command:

    python -m torch.distributed.launch --nproc_per_node 4 train/train.py --gradient_accumulation_steps 4

    python -m torch.distributed.launch  --master_addr *** --master_port *** --nproc_per_node 4 train/train.py --gradient_accumulation_steps 4
