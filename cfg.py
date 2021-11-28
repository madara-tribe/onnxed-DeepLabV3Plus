import os
from easydict import EasyDict

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()
Cfg.num_classes = 5
Cfg.train_batch = 4
Cfg.subdivisions = 1
Cfg.val_batch_size = 1
Cfg.output_stride = 16
Cfg.scheduler_type='polylr'
Cfg.lr = 0.001
Cfg.weight_decay = 1e-4
Cfg.momentum = 0.9
Cfg.total_itrs = 50e3
Cfg.random_seed=1
Cfg.vis_num_samples = 8
Cfg.TRAIN_OPTIMIZER = 'adabelief'
#Cfg.TRAIN_OPTIMIZER = 'adam'
#Cfg.loss_type = 'CrossEntropy'
Cfg.loss_type = 'Focal'
Cfg.gpu_id = '3'
Cfg.vis_port = 13570
Cfg.num_classes = 5
## dataset
Cfg.train_img_dir = '../4cls/train'
Cfg.train_mask_dir = '../4cls/anno'
Cfg.valid_img_dir = '../4cls/val'
Cfg.valid_mask_dir = '../4cls/val_anno'
Cfg.TRAIN_TENSORBOARD_DIR = './logs'
Cfg.ckpt_dir = os.path.join(_BASE_DIR, 'checkpoints')


