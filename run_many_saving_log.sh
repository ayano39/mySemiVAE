action=$1
dev_id=0
dev_num=2
for label_size in 0 100 600 1000 3000 50000
do	
	CUDA_VISIBLE_DEVICES=${dev_id} python -u run_vae.py --action=${action} --num_epoch=100 --lr=0.001 --dim_z=10 --num_layers=1 --num_class_layers=2 --activation=relu --labelled_size=${label_size} > log_${label_size}label &
	dev_id=$(((dev_id+1)%dev_num))
done
	
