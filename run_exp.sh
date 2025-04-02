# !/bin/bash
cd src

torchrun --nproc_per_node 8 \
    -m open_clip_train.main \
    --train-data "/weka/oe-training-default/mm-olmo/torch_datasets/pixmo_datasets/cap/train" \
    --val-data /weka/oe-training-default/mm-olmo/torch_datasets/pixmo_datasets/cap/validation \
    --train-num-samples 714985 \
    --dataset-type "hf" \
    --batch-size 512 \
    --precision amp \
    --workers 8 \
    --epochs 30 \
    --imagenet-val "/weka/prior-default/georges/datasets/imagenet1k/imagenet/val" \
    --model "ViT-L-14-quickgelu" \
    --pretrained "metaclip_fullcc" \
    --lr 5e-6 \
    --wd 0.1 \
    --wandb-project-name metaclip-fullcc-pixmocap-ft \
    --report-to wandb \
    --warmup 2e3
    # --logs /weka/oe_training_default/georges/checkpoints/finetuned_models/metaclip_400m/logs \