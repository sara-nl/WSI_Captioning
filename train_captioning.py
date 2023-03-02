import os
import yaml
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from data.image_text_loader import TextImageDataModule
from models import CaptioningWrapper
from pytorch_lightning.callbacks import LearningRateMonitor

def prepare_callbacks(args):
    
    lr_monitor = LearningRateMonitor(logging_interval="step")

    return [lr_monitor]

def main(args):

    with open(args.captioning_config_dir) as fin:
        capt_config = yaml.safe_load(fin)["Capt"]

    print("capt config: ", capt_config)
    print("args: ", args)
    callbacks = prepare_callbacks(args)

    dm_train = TextImageDataModule.from_argparse_args(args)
    trainer = Trainer.from_argparse_args(args, callbacks=callbacks)
    model = CaptioningWrapper(capt_config, args)
    trainer.fit(model, train_dataloaders=dm_train)

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--captioning_config_dir', type=str, default='models/configs/Capt.yaml')
    parser.add_argument('--vision_lr', type=float, default=5e-5)
    parser.add_argument('--lm_lr', type=float, default=1e-4)
    parser.add_argument('--dataset', default="both", type=str, help='choose between radboud, catania or both')

    parser.add_argument('--lmdb_patches_path', type=str, required=True, help='directory of your training folder')
    parser.add_argument('--text_embeddings_path', type=str, default="/projects/0/examode/caption_generation/colon/embeddings/texts/biogpt")
    parser.add_argument('--text_tokens_path', type=str, default="/projects/0/examode/caption_generation/colon/texts/extracted_tokens/biogpt/token_dict.json")
    parser.add_argument('--crossvalidation_path', type=str, required=True, help='path to the .csv data containing the partitions')
    parser.add_argument('--tokens_path', type=str, default="/projects/0/examode/caption_generation/colon//texts/extracted_tokens/biogpt/token_dict.json")
    parser.add_argument('--val_fold', type=int, default=9, required=True)
    parser.add_argument('--save_generations_path', type=str, required=False, help='path where to save the generations at the last epoch')

    parser.add_argument('--batch_size', type=int, default=1300, help='size of the batch')
    parser.add_argument('--context_length', type=int, default=42, help='context')
    parser.add_argument('--temperature', type=float, default=0.8, help='generation temp if we sample')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the dataloaders')
    parser.add_argument('--shuffle', type=bool, default=True, help='whether to use shuffling during sampling')
    parser.add_argument('--num_samples_val', type=int, default=100)
    parser.add_argument('--clip_checkpoint', type=str, default="./checkpoints/CLIP/exp_21_9_3/epoch_47_step_47.ckpt")
    
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)