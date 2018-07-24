action=$1
labelled_size=$2
CUDA_VISIBLE_DEVICES=1 python -u run_vae.py --action=${action} --num_epoch=100 --lr=0.001 --dim_z=10 --num_layers=1 --num_class_layers=2 --activation=relu --labelled_size=${labelled_size}