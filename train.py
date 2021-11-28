from cfg import Cfg
import torch
from solver import Solver, call_deeplabv3plus_model
import sys, os

def main():
    cfg = Cfg
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.gpu_id
    num_workers = 1 # os.cpu_count()
    weight_path = 'best_os16.pth'
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)
    model = call_deeplabv3plus_model(cfg)
    solvers = Solver(cfg)
    solvers.train(config=cfg,
          model=model,
          device=device,
          num_workers=num_workers,
          pin_memory=True,
          weight_path=weight_path)
        
    
if __name__ == '__main__':
    main()

