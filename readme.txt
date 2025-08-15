train
python train_half_mafunet.py --data_root dataset --img_size 288 384 --epochs 50 --batch_size 8 --amp


eval
python eval_metrics.py --data_root dataset/val --img_size 288 384 --ckpt outputs/checkpoints/best.pt --amp



