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
import glob
import itertools
import json


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
    parser.add_argument('--distance', action='store_true', help='calculate and print distance between DTI images, using the model matching to the case, no training if there is a model ')
    parser.add_argument('--case', help='take the correct corresponding data according to the case argument')
    parser.add_argument('--lambda_distance', default=1, help='lambda for distance between points')
    parser.add_argument('--how_many_points_for_dist', default=50, help='how many points to use foe distance calculate')
    parser.add_argument('--win_len_for_distance', default=5, help='window size to open for each point when calculating distance ')



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
    Animals_to_check = ['Dog',
                        'Hyrax',
                        'WildRat2',
                        'Cow1',
                        'Giraffe1',
                        'Orangutan1',
                        'Donkey',
                        'Chimpanzee',
                        'Horse1',
                        'Wolf1']
    combinations = list(itertools.combinations(Animals_to_check, 2))
    combinations_cases = [f'{pair[0]}_vs_{pair[1]}' for pair in combinations]
    VERBOSE = args.verbose # not used 
    checkpoints_path = '/mnt/storage/datasets/hila_cohen_DTI/outputs/checkpoints'
    case = args.case
    for win_size in [7, 9, 11]:
        distances_dict = {}
        args.win_len_for_distance = win_size
        for case in combinations_cases:
            model_to_load = []
            picked_animals = case.split('_') # you have vs in the middle
            is_only_distance = args.distance #for each case always try to follow the initial intention 
            #look for the model
            if is_only_distance: 
                for subdir, dirs, files in os.walk(checkpoints_path):
                    if picked_animals[0].lower() in subdir.lower() and picked_animals[2].lower() in subdir.lower():
                        model_to_load = glob.glob(f'{subdir}/*best.pth.tar')
                        if len(model_to_load) > 0:
                            break
                if len(model_to_load) == 0:
                    is_only_distance = False # if no model was found, disable distance only and train  
                    model_to_load = [args.load]
            else:
                model_to_load = [args.load] # just train using the input model
                
            with open(args.config) as f:
                cfg = EasyDict(json.load(f))
            # if it is empty, so we won't get exception 
            if len(model_to_load) > 0:
                cfg.load = model_to_load[0]
            else:
                cfg.load = args.load
            cfg.docker = args.docker
            cfg.distance = is_only_distance
            cfg.how_many_points_for_dist = int(args.how_many_points_for_dist)
            cfg.lambda_distance = int(args.lambda_distance)
            cfg.win_len_for_distance = int(args.win_len_for_distance)
            # if we also train
            if is_only_distance:
                cfg.data_case = case + '_only_distance'
            else:
                cfg.data_case = case + '_with_distance'
            
            if args.evaluate or args.test or args.docker or is_only_distance:
                cfg.update({
                    'levels': [1],
                    'epoch_size': -1,
                    'valid_interval': 1,
                    'log_interval': 1,
                })


            # store files day by day
            curr_time = datetime.datetime.now().strftime("%y%m%d%H%M%S")

            cfg.save_root = Path('/mnt/storage/datasets/hila_cohen_DTI/outputs/checkpoints') / curr_time[:6] / (curr_time[6:] + f'_{cfg.data_case}' + f'_{cfg.how_many_points_for_dist}_points' + f'_{cfg.lambda_distance}_lambda' + f'_{cfg.win_len_for_distance}_distance_win_len')

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
            distances_dict[case] = trainer.train(0, 1)

        save_root_json = Path('/mnt/storage/datasets/hila_cohen_DTI/outputs/distances_json') / curr_time[:6] 
        save_root_json.makedirs_p()
        json_name = (curr_time[6:] + f'_{cfg.how_many_points_for_dist}_points' + f'_{cfg.lambda_distance}_lambda' + f'_{cfg.win_len_for_distance}_distance_win_len.json')
        # write all to json 
        with open(os.path.join(save_root_json, json_name), 'w') as fp:
            json.dump(distances_dict, fp)
#! multi GPU 
# mp.spawn(trainer.train,
#           args=(world_size,),
#           nprocs=world_size,
#           join=True)
