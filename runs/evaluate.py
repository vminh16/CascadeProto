""" Prototypical Network for Few-shot 3D Point Cloud Semantic Segmentation [Baseline]
"""
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from runs.training_free import test_few_shot
from dataloaders.loader import MyDataset, MyTestDataset, batch_task_collate
from models.vipseg_learner import Learner
from utils.cuda_util import cast_cuda
from utils.logger import init_logger
from utils.checkpoint_util import load_model_checkpoint

import numpy as np

def print_trainable_params(model_component, logger, component_name):
    # 筛选需要梯度的参数
    trainable_params = filter(lambda p: p.requires_grad, model_component.parameters())
    # 计算参数总数
    total_params = sum(np.prod(p.size()) for p in trainable_params)
    # 打印参数数量，单位为百万（M）
    logger.cprint(f'\nNumber of {component_name} parameters: {total_params / 1e6:.2f} "M"\n')


def eval(args):
    logger = init_logger(args.log_dir, args)
    PL = Learner(args)

    print_trainable_params(PL.model, logger, "All testing")
        
    WRITER = SummaryWriter(log_dir=args.log_dir)
    
    TEST_DATASET = MyTestDataset(args.model, args.data_path, args.dataset, cvfold=args.cvfold,
                                 num_episode_per_comb=args.n_episode_test,
                                 n_way=args.n_way, k_shot=args.k_shot, n_queries=args.n_queries,
                                 num_point=args.pc_npts, pc_attribs=args.pc_attribs, 
                                 way_ratio=args.way_pcratio, way_num=args.way_pcnum, mode='test')
    
    TEST_CLASSES = list(TEST_DATASET.classes)
    TEST_LOADER = DataLoader(TEST_DATASET, batch_size=1, shuffle=False, collate_fn=batch_task_collate)

    PL.model = load_model_checkpoint(args.log_dir)
    test_IoU = test_few_shot(TEST_LOADER, PL, logger, TEST_CLASSES)
    logger.cprint('\n=====[TEST]  Mean IoU: %f =====\n' % (test_IoU))            

    WRITER.close()
