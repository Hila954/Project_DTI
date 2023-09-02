source venv/bin/activate

# baseline - train from cardiac pt
#python main_train.py -c configs/l2r_baseline_server.json -l CardiacChkpts/4DCT_best_w_admm_ckpt.pth.tar -s --cuda_devices 0,1,2,3,4,5,6,7 

# costunrolling - train from cardiac pt
python main_train.py -c configs/l2r_costunrolling_server.json -l CardiacChkpts/4DCT_best_w_admm_ckpt.pth.tar -s --cuda_devices 0,1,2,3,4,5,6,7 
