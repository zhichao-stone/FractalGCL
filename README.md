# FractalGCL

Code of *"Fractal Graph Contrastive Learning"*



### 1. Requirements

```
torch
torch-geometric
scikit-learn
numpy
networkx
```



### 2. How to run

#### step1. pre-process

Run following instruction to get the fractality and fractal dimensions of graphs in dataset.

```shell
CUDA_VISIBLE_DEVICES=[GPU_IDS] python get_gractal_dimensions.py --data TU_DATA[MUTAG, DD, ...]
```

The results will be saved in `./fractal_results/linear_regression_{DATA}.json`. The example is [./fractal_results/linear_regression_mutag.json](./fractal_results/linear_regression_mutag.json)

#### step2. run experiment

```shell
CUDA_VISIBLE_DEVICES=[GPU_IDS] python main_FractalGCL_unsupervised.py \
	--data TU_DATA[MUTAG, DD, ...] \
	--batch_size 128 \
	--aug_type renorm_rc \
	--pretrain_max_epochs 100 \
	--folds 10 \
	--num_repeat_exp 5
```

The detail arguments can be seen in the function  `./utils.py/get_args()`

The results will be saved in `./log/FractalGCL_{DATA}_{AUG_TYPE}.log`



### 3. Experiments for Urban Road Networks

The data, code, and experiment pipeline for urban road networks can be seen under the directory [./FractalGCL_for_city](./FractalGCL_for_city).