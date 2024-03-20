### Installation
```
pip install -r requirements.txt
```

### Data Preparation
We suggest putting all datasets under the same folder (say `$DATA`) to ease management and following the instructions below to organize datasets to avoid modifying the source code. The file structure looks like
```
├── data/
│   └── caltech-101/ 
│   └── dtd/
│   └── eurosat/
│   └── ......
│   └── ucf101/ 
```
If you have some datasets already installed somewhere else, you can create symbolic links in `$DATA/dataset_name` that point to the original data to avoid duplicate download.

### Running Scripts
```
srun -p <partition_name> --mpi=pmi2 --quotatype=auto --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=debug --kill-on-bad-exit=1 \
  sh scripts/zeroshot_topk_save_trainset.sh
```

```
srun -p <partition_name> --mpi=pmi2 --quotatype=auto --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=debug --kill-on-bad-exit=1 \
  sh scripts/zeroshot_topk.sh
```

After running the scripts, three files will be generated:
1. predictions.pth
```
# Picture names of testset: {label, pred_logits, pred_class}
{'/mnt/xxx/data/fgvc_aircraft/images/1514522.jpg':
  {'label': tensor(0),
   'pred_logits': tensor([31.6562, 31.1875, 30.9531, 30.7344, 30.5625, 30.4219, 30.3594, 30.3125,
        30.2969, 30.2344], dtype=torch.float16),
   'pred_class': tensor([22, 21, 15, 14, 17, 16, 18, 30, 29, 32])}
}
```

2. classnames.pth
```
['707-320', '727-200', '737-200', '737-300', '737-400', ...]
```

3. trainset.pth
```
# Picture names of trainset：class label
{'/mnt/xxx/data/fgvc_aircraft/images/1025794.jpg': {'label': 0}}
```

### Read predictions.pth，obtain Top-1/Top-5 Accuracy

```
srun -p <partition_name> --mpi=pmi2 --quotatype=auto --gres=gpu:1 -n1 --ntasks-per-node=1 --job-name=debug --kill-on-bad-exit=1 \
  sh scripts/zeroshot_topk_eval.sh
```

The expected output is as follows：
```
=> result
* total: 3,333
* top1_accuracy: 24.69%
* top10_accuracy: 77.77%
```
