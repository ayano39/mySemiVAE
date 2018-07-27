action=$1
dev_id=0
dev_num=2

unlabelled_size=40000
for labelled_size in 0 10 100 1000 5000
do
	CUDA_VISIBLE_DEVICES=${dev_id} nohup bash run_one.sh ${action} ${labelled_size} ${unlabelled_size} > log_${labelled_size}l_${unlabelled_size}u.txt &
	dev_id=$(((dev_id+1)%dev_num))
done
