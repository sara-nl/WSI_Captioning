# WSI_captioning
Repo for WSI caption generation

# Requirements

module load 2022\
module load Python/3.10.4-GCCcore-11.3.0\
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0\
pip install transformers\
pip install pytorch-lightning\
pip install python-dateutil\

*NOTE: PyTorch 1.12&1.13 are SLOWER than 1.11 with CUDA 11.3.1; hypothesis: it's actually Lightning's dataloading.

# Run 
./train_captioning.sh
