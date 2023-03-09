# WSI_captioning
Repo for WSI caption generation

# Requirements

```
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
pip install transformers
pip install pytorch-lightning
pip install python-dateutil 
```

*NOTE: PyTorch 1.12&1.13 are SLOWER than 1.11 with CUDA 11.3.1; hypothesis: it's actually Lightning's dataloading.

# Run 

Make sure to put the data, i.e. ``/projects/0/examode/caption_generation/colon``, to scratch for fast reading. Just one GPU should be good enough. Training to 80 epochs takes around 10 mins for me on a gcn node. Also note, the workers for the dataloading are kinda broken, so training should be faster than this.

Change the arguments as seen fit in train_captioning.sh and then run
```
sh train_captioning.sh
```

This trains the model but does not evaluate it during training apart from computing the validation loss.

To evaluate the trained model you need to change the ``--load_from_checkpoint`` path with the newly trained model.
For example:

```
--load_from_checkpoint ${root_data}/checkpoints/CAPT/exp_28/epoch_149_step_750.ckpt
```

Thereafter, evaluate the trained model with:

```
sh run_eval.sh
```