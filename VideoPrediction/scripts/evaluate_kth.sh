# KTH
gpu_no=0

dataset=kth
log_root=kth
checkpoint=240000
method_dir=ours_gan

CUDA_VISIBLE_DEVICES=${gpu_no} python scripts/evaluate_ours.py --input_dir data/${dataset} --dataset_hparams sequence_length=30 --checkpoint logs/${log_root}/${method_dir}/model-${checkpoint} --mode test --results_dir eval_results/${log_root}/${method_dir}/${checkpoint} --batch_size 1 --only_metrics
