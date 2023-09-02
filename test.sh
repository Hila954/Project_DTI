source venv/bin/activate

#python main_train.py -t -c configs/l2r_costunrolling.json -l checkpoints/l2r_4dct_costunrolling_cyc_0.01_model_best.pth.tar

# test
#python main_train.py -t -c configs/l2r_costunrolling.json -l checkpoints/l2r_4dct_costunrolling_mwl_cyc_0.01_model_best.pth.tar
python main_train.py -t -c configs/l2r_costunrolling_test.json -l checkpoints/l2r_4dct_costunrolling_mwl_cyc_0.01_model_best.pth.tar
python main_train.py -t -c configs/l2r_costunrolling_test.json -l checkpoints/l2r_4dct_baseline_mwl_wo_gaus_model_best.pth.tar


# docker
#python main_train.py -d -c configs/l2r_costunrolling_docker.json -l checkpoints/l2r_4dct_costunrolling_mwl_cyc_0.01_model_best.pth.tar

#python main_train.py -t -c configs/l2r_baseline.json -l checkpoints/l2r_4dct_baseline_mwl_wo_gaus_model_best.pth.tar
