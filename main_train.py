import torch
# from utils.torch_utils import init_seed
import argparse
import datetime
from path import Path
from data.dataset import get_dataset
from models.pwc3d import get_model
from losses.flow_loss import get_loss
from trainer.get_trainer import get_trainer
import json
import os
from easydict import EasyDict
import torch.multiprocessing as mp


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='4DCT Optical Flow Net')
    parser.add_argument('-c', '--config', default='configs/l2r_costunrolling.json', help="path for config file")
    parser.add_argument('-v', '--verbose', action='store_true', help="Verbose logs")
    parser.add_argument('-p', '--plot', action='store_true', help="Plot samples along training")
    parser.add_argument('-l', '--load', help="Model weights *.pth.tar file")
    parser.add_argument('-e', '--evaluate', action='store_true', help="run eval only")
    parser.add_argument('-t', '--test', action='store_true', help="run test (dump disp files)")
    parser.add_argument('-d', '--docker', action='store_true', help="run test (dump disp files)")
    parser.add_argument('-s', '--server', action='store_true', help='run on sever')
    parser.add_argument('--distance', action='store_true', help='calculate and print distance between DTI images, using the loaded model ')

    parser.add_argument('--cuda_devices', default='0,1', help="visible cuda devices")

    args = parser.parse_args()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_devices

    # models_to_load = ["outputs/checkpoints/240112/225640_lr_0.0001_from_scratch_no_resample/model_DTI_DOG_HYRAX_yariv_model_best.pth.tar",
    #                     "outputs/checkpoints/240202/162434_lr_0.0001_finetune_smoothness_high_occ_threshold/model_DTI_DOG_HYRAX_model_best.pth.tar",
    #                     "outputs/checkpoints/240224/152426_lr_0.0001_smoothness_0.05_smallest_level/model_DTI_DOG_HYRAX_model_best.pth.tar",
    #                     "outputs/checkpoints/240224/153020_lr_0.0001_smoothness_0.01_biggest_level/model_DTI_DOG_HYRAX_model_best.pth.tar",
    #                     "outputs/checkpoints/240225/232035_lr_0.0001_0.05_sm_smallest_two_levels/model_DTI_DOG_HYRAX_model_best.pth.tar",
    #                     "outputs/checkpoints/240225/232140_lr_0.0001_0.01_sm_smallest_two_levels/model_DTI_DOG_HYRAX_model_best.pth.tar",
    #                     "outputs/checkpoints/240229/231322_lr_0.0001_last_two_levels_0.1/model_DTI_DOG_HYRAX_model_best.pth.tar",
    #                     "outputs/checkpoints/240229/232731_lr_0.0001_all_levels_smothness_0.01/model_DTI_DOG_HYRAX_model_best.pth.tar",
    #                     "outputs/checkpoints/240309/163305_lr_0.0001_all_levels_smothness_0.1/model_DTI_DOG_HYRAX_model_best.pth.tar"]
    VERBOSE = args.verbose
    load = args.load
    with open(args.config) as f:
        cfg = EasyDict(json.load(f))
    cfg.load = load
    cfg.docker = args.docker
    cfg.distance = args.distance
    
    if args.evaluate or args.test or args.docker or args.distance:
        cfg.update({
            'levels': [1],
            'epoch_size': -1,
            'valid_interval': 1,
            'log_interval': 1,
        })

    # if args.test or args.docker:        
    #     cfg.update({
    #         'dump_disp': True,
    #     })

    # store files day by day
    curr_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")
    if args.server:
        cfg.save_root = Path('/mnt/storage/datasets/glifshitz_user_data/4dct/outputs/checkpoints') / curr_time[:6] / curr_time[6:]
    else:
        cfg.save_root = Path('./outputs/checkpoints') / curr_time[:6] / f'{curr_time[6:]}_lr_{cfg.lr}'
    
    if args.docker:
        cfg.save_root = Path('docker_submission')

    cfg.save_root.makedirs_p()

    train_set = get_dataset(cfg, valid=False, root=cfg.data_path, w_aug=True, data_type=cfg.train_type, frame_dif=cfg.frame_dif)
    if args.docker:
            valid_set = get_dataset(cfg, valid=True, root=cfg.valid_path, w_aug=False, data_type='l2r_test')
    else:
        valid_set = get_dataset(cfg, valid=True, root=cfg.valid_path, w_aug=False, data_type=cfg.valid_type)
    model = get_model(cfg)
    loss = get_loss(cfg)
    
    trainer = get_trainer()(
        train_set, valid_set, model, loss, cfg 
    )

    # run DDP
    
    world_size = torch.cuda.device_count()
    trainer.train(0, 1)
    #! GPU 
    # mp.spawn(trainer.train,
    #           args=(world_size,),
    #           nprocs=world_size,
    #           join=True)
