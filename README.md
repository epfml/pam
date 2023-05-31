# Hardware-Efficient Transformer Training via Piecewise Affine Operations
This is the official code for our [preprint](https://arxiv.org/abs/2305.17190), where we investigate the use of cheap piecewise affine alternatives to common neural network operations such as multiplications for hardware-efficient training.
The abstract is repeated below:

Multiplications are responsible for most of the computational cost involved in neural network training and inference. Recent research has thus looked for ways to reduce the cost associated with them. Inspired by [Mogami (2020)](https://arxiv.org/abs/2012.03458), we replace multiplication with a cheap piecewise affine approximation that is achieved by adding the bit representation of the floating point numbers together as integers. We show that transformers can be trained with the resulting modified matrix multiplications on both vision and language tasks with little to no performance impact, and without changes to the training hyperparameters. We further replace all non-linearities in the networks making them fully and jointly piecewise affine in both inputs and weights. Finally, we show that we can eliminate all multiplications in the entire training process, including operations in the forward pass, backward pass and optimizer update, demonstrating the first successful training of modern neural network architectures in a fully multiplication-free fashion.

## Setup
We use modified versions of two existing libraries [TIMM](https://huggingface.co/docs/timm/index) and [FairSeq](https://github.com/facebookresearch/fairseq) for our baseline, found in the submodules directory. The changes focus on allowing us to replace layers and operations. The shared folder contains code to integrate those modules with our implementation. The piecewise affine operations are implemented in the pam subdirectory (for **P**iecewise **A**ffine **M**ultiplication).

The submodules have their own setup instructions, our kernels additionally require the cuda-devkit and ninja compiler.
We use the following environment setup for a minimal working example:

```
conda create -n pam python=3.10
conda activate pam
conda install ninja cudatoolkit-dev pytest
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyyaml wandb 
```

After this the PAM tests should pass (assumes a CUDA enabled GPU is available):
```
cd pam
pytest
```

## Training
Below we list example commands that can be used to replicate experiments in our manuscript.
The environmental variables, e.g. DATA_ROOT etc, should be adjusted as needed.
The BASE_DIR should be this current root directory.

### FAIRSEQ IWSLT2014-De-En
```
# PATHS
BASE_DIR=$(pwd)
FAIRSEQ_ROOT=$BASE_DIR/submodules/fairseq
DATA_ROOT=/home/$USER/datasets
IWSLT14=$DATA_ROOT/iwslt14

# Install fairseq
cd $FAIRSEQ_ROOT
pip install --editable ./

# Download and prepare the data
mkdir $IWSLT14
cd $IWSLT14
bash $FAIRSEQ_ROOT/examples/translation/prepare-iwslt14.sh
cd $BASE_DIR
fairseq-preprocess --source-lang de --target-lang en \
    --trainpref $IWSLT14/iwslt14.tokenized.de-en/train --validpref $IWSLT14/iwslt14.tokenized.de-en/valid --testpref $IWSLT14/iwslt14.tokenized.de-en/test \
    --destdir $IWSLT14/bin/iwslt14.tokenized.de-en \
    --workers 20

# Train model
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PYTHONPATH:$BASE_DIR fairseq-train \
    $IWSLT14/bin/iwslt14.tokenized.de-en \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler cosine --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 4096 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --linear-cfg standard \
    --max-update 22021 \
    --save-dir runs/baseline20_cosine \
    --wandb-project fairseq \

# Extra arguments if needed (e.g. to replace operations with their piecewise alternatives):
--wandb-project YOUR_PROJECT_NAME
--log-file runs/baseline20/log.txt
--linear-cfg '{"subtype": "pam", "dkwargs": {"approx_bwd": true}}'
--norm-cfg '{"subtype": "pam_ln", "dkwargs": {"approx_bwd": true}}'
--functional-cfg '{"type": "pam",  "dkwargs": {"approx_bwd": true}}'
--loss-functional-cfg '{"type": "pam",  "dkwargs": {"approx_bwd": true}}'
--opt-functional-cfg pam
```

### TIMM DeiT CIFAR10 & i1k
```
# PATHS
BASE_DIR=$(pwd)
cd submodules/timm
OUT_DIR=/home/$USER/runs/timm/
DATA_DIR=/home/$USER/datasets
WANDB_PROJECT=PROJECT_NAME

PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 4 train.py --model deit_tiny_patch16_224 --output $OUT_DIR --data-dir $DATA_DIR --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 224 224 --color-jitter 0.3 --aa rand-m9-mstd0.5-inc1 --smoothing 0.1 --train-interpolation bicubic --mixup 0.8 --cutmix 1.0 --reprob 0.25 --drop-path 0.1 -b 64 --opt adamw --lr 5e-4 --opt-eps 1e-8 --weight-decay 0.05 --sched cosine --epochs 600 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --log-wandb --wandb-kwargs project=$WANDB_PROJECT name=baseline_deit_tiny_cifar10

PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 4 train.py --model deit_tiny_patch16_224 --output $OUT_DIR --data-dir $DATA_DIR --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 224 224 --color-jitter 0.3 --aa rand-m9-mstd0.5-inc1 --smoothing 0.1 --train-interpolation bicubic --mixup 0.8 --cutmix 1.0 --reprob 0.25 --drop-path 0.1 -b 64 --opt adamw --lr 5e-4 --opt-eps 1e-8 --weight-decay 0.05 --sched cosine --epochs 600 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --log-wandb --wandb-kwargs project=$WANDB_PROJECT name=pam_deit_tiny_cifar10 --model-kwargs linear_cfg='{"subtype":"pam","dkwargs":{"approx_bwd":True}}' conv_cfg='{"subtype":"pam","dkwargs":{"approx_bwd":True}}' bmm_cfg='{"type":"pam","dkwargs":{"approx_bwd":True}}'

PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 6 train.py --model deit_tiny_patch16_224 --output $OUT_DIR --data-dir $DATA_DIR/imagenet --dataset ImageFolder --num-classes 1000 --pin-mem --input-size 3 224 224 --workers 16 --color-jitter 0.3 --aa rand-m9-mstd0.5-inc1 --smoothing 0.1 --train-interpolation bicubic --mixup 0.8 --cutmix 1.0 --reprob 0.25 --drop-path 0.1 -b 192 --opt adamw --lr 5e-4 --lr-base-size 512 --opt-eps 1e-8 --weight-decay 0.05 --sched cosine --sched-on-update --epochs 300 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --log-wandb --wandb-kwargs project=$WANDB_PROJECT name=pam_deit_tiny_i1k --model-kwargs linear_cfg='{"subtype":"pam","dkwargs":{"approx_bwd":True}}' conv_cfg='{"subtype":"pam","dkwargs":{"approx_bwd":True}}' bmm_cfg='{"type":"pam","dkwargs":{"approx_bwd":True}}'

PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 6 train.py --model deit_tiny_patch16_224 --output $OUT_DIR --data-dir $DATA_DIR/imagenet --dataset ImageFolder --num-classes 1000 --pin-mem --input-size 3 224 224 --workers 16 --color-jitter 0.3 --aa rand-m9-mstd0.5-inc1 --smoothing 0.1 --train-interpolation bicubic --mixup 0.8 --cutmix 1.0 --reprob 0.25 --drop-path 0.1 -b 192 --opt adamw --lr 5e-4 --lr-base-size 512 --opt-eps 1e-8 --weight-decay 0.05 --sched cosine --sched-on-update --epochs 300 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --log-wandb --wandb-kwargs project=$WANDB_PROJECT name=baseline_deit_tiny_i1k
```

### TIMM CIFAR10 CNNs
Baseline commands are given below, add the following `linear_cfg='{"subtype":"pam","dkwargs":{"approx_bwd":True}}' conv_cfg='{"subtype":"pam","dkwargs":{"approx_bwd":True}}'` to `--model-kwargs` to use PAM matmuls and convolutions.
```
# PATHS
BASE_DIR=$(pwd)
cd submodules/timm
OUT_DIR=/home/$USER/runs/timm/
DATA_DIR=/home/$USER/datasets
WANDB_PROJECT=PROJECT_NAME

# VGG13 without BN, num_features=512
PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $OUT_DIR \
--data-dir $DATA_DIR --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --crop-pct 1 --random-crop-pad 4 --color-jitter 0.0 --smoothing 0.0 \
--model vgg13 --model-kwargs num_features=512 feature_window_size=1 \
-b 128 --opt sgd --lr 0.05 --momentum 0.9 --weight-decay 5e-4 --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
--log-wandb --wandb-kwargs project=$WANDB_PROJECT name=vgg13_c10_baseline

# Pre-residual RN20
PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $OUT_DIR \
--data-dir $DATA_DIR --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --crop-pct 1 --random-crop-pad 4 --color-jitter 0.0 --smoothing 0.0 \
--model cifar_resnet --model-kwargs name=cifar_pre_resnet20 \
-b 256 --opt sgd --lr 0.2 --momentum 0.9 --weight-decay 2e-4 --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
--log-wandb --wandb-kwargs project=$WANDB_PROJECT name=rn20_c10_baseline

# Pre-residual RN110
PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $OUT_DIR \
--data-dir $DATA_DIR --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --crop-pct 1 --random-crop-pad 4 --color-jitter 0.0 --smoothing 0.0 \
--model cifar_resnet --model-kwargs name=cifar_pre_resnet110 \
-b 256 --opt sgd --lr 0.2 --momentum 0.9 --weight-decay 2e-4 --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
--log-wandb --wandb-kwargs project=$WANDB_PROJECT name=rn110_c10_baseline

# RNXT20_4x16
PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $OUT_DIR \
--data-dir $DATA_DIR --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --crop-pct 1 --random-crop-pad 4 --color-jitter 0.0 --smoothing 0.0 \
--model cifar_resnet --model-kwargs block_name=Bottleneck stage_depths=[2,2,2] input_block=cifar base_channels=64 input_block_channels=16 cardinality=4 pre_activation=False \
-b 128 --opt sgd --lr 0.1 --momentum 0.9 --weight-decay 5e-4 --sched cosine --sched-on-update --epochs 200 --warmup-lr 1e-6 --min-lr 1e-5 --warmup-epochs 5 --checkpoint-hist 1 \
--log-wandb --wandb-kwargs project=$WANDB_PROJECT name=rnxt20_4x16_c10_baseline

# ConvMixer 256/8
PYTHONPATH=$PYTHONPATH:$BASE_DIR torchrun --nproc_per_node 1 train.py --output $OUT_DIR --checkpoint-hist 1 \
--model convmixer_cifar --model-kwargs kernel_size=5 patch_size=2 \
--data-dir $DATA_DIR --dataset torch/CIFAR10 --dataset-download --num-classes 10 --pin-mem --input-size 3 32 32 --mean 0.4914 0.4822 0.4465 --std 0.2023 0.1994 0.2010 --no-prefetcher \
--train-transforms RandomResizedCrop='{"size":32,"scale":[1.0,1.0],"ratio":[1.0,1.0]}' RandomHorizontalFlip='{"p":0.5}' RandAugment='{"num_ops":2,"magnitude":12}' ColorJitter='{"brightness":0.1,"contrast":0.1,"saturation":0.1}' ToTensor='{}' Normalize='{"mean":[0.4914,0.4822,0.4465],"std":[0.2023,0.1994,0.2010]}' RandomErasing='{"p":0.2}' \
--test-transforms ToTensor='{}' Normalize='{"mean":[0.4914,0.4822,0.4465],"std":[0.2023,0.1994,0.2010]}' \
-b 128 --opt adamw --lr 0.01 --weight-decay 5e-2 --sched cosine --sched-on-update --epochs 100 --warmup-epochs 5 --opt-eps=1e-3 \
--log-wandb --wandb-kwargs project=$WANDB_PROJECT name=convmixer_c10_long_baseline

```

## Citation
If you use this work, please consider citing us using the following:
```
@misc{kosson2023hardwareefficient,
      title={Hardware-Efficient Transformer Training via Piecewise Affine Operations}, 
      author={Atli Kosson and Martin Jaggi},
      year={2023},
      eprint={2305.17190},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```