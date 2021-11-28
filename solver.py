from tqdm import tqdm
import os
import random
import numpy as np
from torch.utils import data
import torch
from torchsummary import summary
import albumentations as albu
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter

import matplotlib
import matplotlib.pyplot as plt
import cv2
from PIL import Image

from visualizer import Visualizer
from losses.multi import MultiClassCriterion
from data_utils.optimizer import create_optimizer
from data_utils.stream_metrics import StreamSegMetrics, AverageMeter
from data_utils.data_loader import SegmentDataLoader
from data_utils.utils import set_bn_momentum
from network.modeling import deeplabv3plus_mobilenet
from cfg import Cfg
crop_size = 35
W = int(1216/4)-crop_size
H = int(1936/4)
    

def get_loader(config, num_worker, pin_memory=True):
    """ Dataset And Augmentation
    https://qiita.com/kurilab/items/b69e1be8d0224ae139ad
    """
    train_transform = albu.Compose([
            albu.HorizontalFlip(p=1),
            #albu.MultiplicativeNoise(),
            #albu.RandomSnow(),
            #albu.RandomShadow(),
            ToTensorV2(),
            ])

    val_transform = albu.Compose([
            ToTensorV2(),
            ])

    train_dst = SegmentDataLoader(config.train_img_dir, config.train_mask_dir, width=W, height=H, transform=train_transform, gammas=3.0, crop=crop_size)
    val_dst = SegmentDataLoader(config.valid_img_dir, config.valid_mask_dir, width=W, height=H, transform=val_transform, gammas=3.0, crop=crop_size)
    
    trainLoader = data.DataLoader(
            train_dst, batch_size=config.train_batch, shuffle=True, num_workers=num_worker, pin_memory=pin_memory)
    valLoader = data.DataLoader(
                val_dst, batch_size=config.val_batch_size, shuffle=True, num_workers=num_worker, pin_memory=pin_memory)
    print("Train set: %d, Val set: %d" %(len(train_dst), len(val_dst)))
    return trainLoader, valLoader, int(len(val_dst))
    
def call_deeplabv3plus_model(config):
    model_map = {
        #'deeplabv3_resnet50': deeplabv3_resnet50,
        #'deeplabv3plus_resnet50': deeplabv3plus_resnet50,
        #'deeplabv3_resnet101': deeplabv3_resnet101,
        #'deeplabv3plus_resnet101': deeplabv3plus_resnet101,
        #'deeplabv3_mobilenet': deeplabv3_mobilenet,
        'deeplabv3plus_mobilenet': deeplabv3plus_mobilenet
    }
    model = model_map['deeplabv3plus_mobilenet'](num_classes=config.num_classes, output_stride=config.output_stride)
    return model
    
class Solver:
    def __init__(self, config):
        self.initial(config)
        os.makedirs(config.ckpt_dir, exist_ok=True)
        
    def initial(self, config):
        print('random seed')
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        random.seed(config.random_seed)
        
        if config.loss_type == 'Focal':
            self.criterion = MultiClassCriterion(loss_type='Focal', ignore_index=255, size_average=True)
        elif config.loss_type == 'CrossEntropy':
            self.criterion = MultiClassCriterion(loss_type='CrossEntropy', ignore_index=255, reduction='mean')
        
        self.tfwriter = SummaryWriter(log_dir=config.TRAIN_TENSORBOARD_DIR)
        
    def save_ckpt(self, path, cur_itrs, optimizer, scheduler, best_score, model):
        """ save current model
        """
        torch.save({
            "cur_itrs": cur_itrs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "best_score": best_score,
        }, path)
        print("Model saved as %s" % path)
        
    def validate(self, model, loader, device, metrics, ret_samples_ids=None):
        """Do validation and return specified samples"""
        metrics.reset()
        ret_samples = []
        save_val_results = True
        interval_valloss = 0
        if save_val_results:
            if not os.path.exists('results'):
                os.mkdir('results')
            img_id = 0

        with torch.no_grad():
            for i, (images, labels) in tqdm(enumerate(loader)):
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                outputs = model(images)
                
                # val loss
                val_loss = self.criterion(outputs, labels)
                val_nploss = val_loss.detach().cpu().numpy()
                interval_valloss += val_nploss
                
                preds = outputs.detach().max(dim=1)[1].cpu().numpy()
                targets = labels.cpu().numpy()

                metrics.update(targets, preds)
                if ret_samples_ids is not None and i in ret_samples_ids:  # get vis samples
                    ret_samples.append(
                        (images[0].detach().cpu().numpy(), targets[0], preds[0]))

                if save_val_results:
                    for i in range(len(images)):
                        image = images[i].detach().cpu().numpy()
                        target = targets[i]
                        pred = preds[i]

                        image = (image * 255).transpose(1, 2, 0).astype(np.uint8)
                        target = loader.dataset.decode_target(target).astype(np.uint8)
                        pred = loader.dataset.decode_target(pred).astype(np.uint8)
                        #plt.imshow(pred),plt.show()
                        pred = np.hstack([image, target, pred])
                        #Image.fromarray(image).save('results/%d_image.png' % img_id)
                        #Image.fromarray(target).save('results/%d_target.png' % img_id)
                        Image.fromarray(pred).save('results/%d_pred.png' % img_id)
                
                        fig = plt.figure()
                        plt.imshow(image)
                        plt.axis('off')
                        plt.imshow(pred, alpha=0.7)
                        ax = plt.gca()
                        ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
                        ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
                        plt.savefig('results/%d_overlay.png' % img_id, bbox_inches='tight', pad_inches=0)
                        plt.close()
                        img_id += 1

            score = metrics.get_results()
        val_losses = interval_valloss
        interval_valloss = 0
        return score, ret_samples, val_losses
        
    def train(self, config, model, device, num_workers, pin_memory, weight_path=None):
        lr = config.lr
        enable_vis = False
        vis_env = 'main'
        vis_port = config.vis_port
        vis = Visualizer(port=vis_port, env=vis_env) if enable_vis else None
        num_classes = config.num_classes
        output_stride = config.output_stride
        vis_num_samples = config.vis_num_samples
        best_score = 0.0
        cur_itrs = 0
        cur_epochs = 0
        vis_sample_id = np.random.randint(0, len(val_loader), vis_num_samples,
                                        np.int32) if enable_vis else None
        interval_loss = 0
        val_loss = 0
        val_interval = 400
        total_itrs = config.total_itrs
        separable_conv = False
        
        print('loading data....')
        train_loader, val_loader, numval = get_loader(config, num_workers, pin_memory=pin_memory)
        
        if separable_conv and 'plus' in 'deeplabv3plus_mobilenet':
            convert_to_separable_conv(model.classifier)
        set_bn_momentum(model.backbone, momentum=0.01)
        metrics = StreamSegMetrics(num_classes)

        print('Set up optimizer and model......')
        optimizer, scheduler = create_optimizer(model, config)

        # Restore
        if weight_path is not None:
            model.load_state_dict(torch.load(weight_path)['model_state'])
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)
        
        #==========   Train Loop   ==========#
        while True: # cur_itrs < config.total_itrs:
            # =====  Train  =====
            model.train()
            cur_epochs += 1
            for (images, labels) in train_loader:
                cur_itrs += 1
                images = images.to(device, dtype=torch.float32)
                labels = labels.to(device, dtype=torch.long)
                optimizer.zero_grad()
                outputs = model(images)
                loss = self.criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                
                np_loss = loss.detach().cpu().numpy()
                interval_loss += np_loss

                if (cur_itrs) % 10 == 0:
                    self.tfwriter.add_scalar('train/train_loss', loss.item(), cur_itrs)
                    self.tfwriter.add_scalar('train/interval_loss', interval_loss/10, cur_itrs)
                    interval_loss = interval_loss/10
                    print("Epoch %d, Itrs %d/%d, Loss=%f" %
                        (cur_epochs, cur_itrs, total_itrs, interval_loss))
                    interval_loss = 0.0
                    
                if (cur_itrs) % val_interval == 0:
                    self.save_ckpt('checkpoints/latest_deeplabv3plus_os{}d.pth'.format(cur_itrs), cur_itrs, optimizer, scheduler, best_score, model)
                    print("validation...")
                    model.eval()
                    val_score, ret_samples, interval_valloss = self.validate(
                        model=model, loader=val_loader, device=device, metrics=metrics, ret_samples_ids=vis_sample_id)
                    print(metrics.to_str(val_score))

                    if val_score['Mean IoU'] > best_score:  # save best model
                        best_score = val_score['Mean IoU']
                        self.save_ckpt('checkpoints/best_os%d.pth' %
                                (output_stride), cur_itrs, optimizer, scheduler, best_score, model)

                    if vis is not None:  # visualize validation score and samples
                        vis.vis_scalar("[Val] Overall Acc", cur_itrs, val_score['Overall Acc'])
                        vis.vis_scalar("[Val] Mean IoU", cur_itrs, val_score['Mean IoU'])
                        vis.vis_table("[Val] Class IoU", val_score['Class IoU'])
                        self.tfwriter.add_scalar('val/acc', val_score['Overall Acc'], cur_itrs)
                        self.tfwriter.add_scalar('val/meanIou', val_score['Mean IoU'], cur_itrs)
                        self.tfwriter.add_scalar('val/clsIou', val_score['Class IoU'], cur_itrs)
                        self.tfwriter.add_scalar('val/val_loss', interval_valloss/numval, cur_itrs)
                        
                        for k, (img, target, lbl) in enumerate(ret_samples):
                            img = (img * 255).astype(np.uint8)
                            target = train_dst.decode_target(target).transpose(2, 0, 1).astype(np.uint8)
                            lbl = train_dst.decode_target(lbl).transpose(2, 0, 1).astype(np.uint8)
                            concat_img = np.concatenate((img, target, lbl), axis=2)  # concat along width
                            vis.vis_image('Sample %d' % k, concat_img)
                       
                    model.train()
                scheduler.step()
                if cur_itrs >= total_itrs:
                    break
        self.tfwriter.close()

