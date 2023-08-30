# WSI_captioning
Repo for WSI caption generation

# Requirements

```
module load 2021
module load Python/3.9.5-GCCcore-10.3.0
module load PyTorch/1.11.0-foss-2021a-CUDA-11.3.1
module unload typing-extensions/3.10.0.0-GCCcore-10.3.0

python -m venv venv_1.11
source venv_1.11/bin/activate

pip install transformers
pip install pytorch-lightning
pip install pytz
pip install python-dateutil 
```

*NOTE: PyTorch 1.12&1.13 are SLOWER than 1.11 with CUDA 11.3.1; hypothesis: it's actually Lightning's dataloading.

# Run 

Make sure to copy the data to your scratch for fast reading. Using just one GPU should be sufficient; remember, we don't have that much data.

 Training to 80 epochs takes less than 10 mins for me on a gcn node. Also note, the workers for the dataloading are kinda broken for now, so training should be faster than this, but that's a TODO.

The default data for running the code is currently under

```
root_data=/projects/0/examode/caption_generation/colon
```

You need to copy the following folders and files to, for instance, your ``/scratch-local/user/caption_generation/``:

1. **The visual embeddings**: ${root_data}/embeddings/hipt/hipt_wsi_embeddings/
2. **The cross validation split file**: ${root_data}/texts/cross_validation_folds/10_cross_validation.csv
3. **Microsoft's Bio-GPT embeddings**: ${root_data}/embeddings/texts/biogpt
4. **The dictionary WSI_name -> tokenized_reports**: ${root_data}/texts/extracted_tokens/biogpt/token_dict.json
5. **Pre-Trained CLIP model checkpoint**: ${root_data}/checkpoints/CLIP/exp_39_9/epoch_59_step_59.ckpt

This will not take too long as it is little under 4G of data.

*NOTE: I will make a bash script for this soon enough.

Next, clone the repo into your scratch-local/user under ``caption_generation``.

Change the arguments as you see fit in train_captioning.sh. But make sure that you modify ``root_data`` first. 

```
root_data=../colon
```

and then run

```
sh train_captioning.sh
```

This trains the model but does not evaluate it during training apart from computing the validation loss.

To evaluate the trained model you need to change the ``--load_from_checkpoint`` path with the newly trained model in ``run_eval.sh``.

For example:

```
--load_from_checkpoint ${root_data}/checkpoints/CAPT/exp_28/epoch_149_step_750.ckpt
```

Also modify ``root_data`` in ``run_eval.sh``, as seen above.


Thereafter, evaluate the trained model with:

```
sh run_eval.sh
```

The mean validation results will be printed on screen.

# Acknowledgement
This project has received funding from the European Union’s Horizon 2020 research and innovation programme under grant agreement No 825292. This project is better known as the ExaMode project. The objectives of the ExaMode project are:
1. Weakly-supervised knowledge discovery for exascale medical data.  
2. Develop extreme scale analytic tools for heterogeneous exascale multimodal and multimedia data.  
3. Healthcare & industry decision-making adoption of extreme-scale analysis and prediction tools.

For more information on the ExaMode project, please visit www.examode.eu. 

![enter image description here](https://www.examode.eu/wp-content/uploads/2018/11/horizon.jpg)  ![enter image description here](https://www.examode.eu/wp-content/uploads/2018/11/flag_yellow.png) <img src="https://www.examode.eu/wp-content/uploads/2018/11/cropped-ExaModeLogo_blacklines_TranspBackGround1.png" width="80">
