import os
import torch
from pathlib import Path
from torch import Tensor
from typing import Tuple
from .captioner import WSICaptioner, CLIP
from argparse import ArgumentParser

from argparse import ArgumentParser

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parser(trainer):
    parser = ArgumentParser()
    parser.add_argument('--captioning_config_dir', type=str, default='models/configs/Capt.yaml')
    parser.add_argument('--lm_lr', type=float, default=1e-4)

    parser.add_argument('--wsi_embeddings_path', type=str, required=True, help="./data/embeddings/wsi_embeddings/")
    parser.add_argument('--text_embeddings_path', type=str, default="./data/embeddings/text_embeddings/")
    parser.add_argument('--tokens_path', type=str, default="./data/embeddings/tokens/")

    parser.add_argument('--batch_size', type=int, default=1300, help='size of the batch')
    parser.add_argument('--context_length', type=int, default=42, help='context')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for the dataloaders')
    parser.add_argument('--shuffle', type=bool, default=True, help='whether to use shuffling during sampling')
    parser.add_argument('--num_samples_val', type=int, default=100)
    parser.add_argument('--clip_checkpoint', type=str, default="./checkpoints/CLIP/epoch_47_step_47.ckpt")
    
    parser = trainer.add_argparse_args(parser)
    args = parser.parse_args()
    return args

def ranked_cosine_similarity(tensor_1: Tensor, tensor_2: Tensor) -> Tuple[Tensor]:
    """
    Compute cosine similarity between two tensors and rank it according to the similarity.
    """
    dot_similarity = tensor_1 @ tensor_2.t()
    sorted, indices = torch.topk(dot_similarity, k=dot_similarity.shape[1], dim=1)
    indices = indices.cpu().numpy()

    # Get the best cosine similarities in the batch 
    #best_similarity = sorted[0,:].cpu().numpy().mean()

    return sorted, indices

def load_CLIP(capt_config, args):
    clip = CLIP(embed_dim=capt_config["embed_dim"], context_length=capt_config["context_length"])
    clip.to(device)
    clip.eval()

    # Load CLIP model in parts
    state_dict = torch.load(args.clip_model_path, map_location=torch.device("cpu"))["state_dict"]
    print([key for key, value in state_dict.items()])
    state_dict_visual = {key.split("model._orig_mod.visual.")[1]: value for key, value in state_dict.items() if "visual" in key}
    clip.visual.load_state_dict(state_dict_visual)

    state_dict_text_projection = {key.split("model._orig_mod.text_projection.")[1]: value for key, value in state_dict.items() if "text_projection" in key}
    clip.text_projection.load_state_dict(state_dict_text_projection)

    return clip

def load_LM_model(capt_config, args):

    model = WSICaptioner(**capt_config)
    model.to(device)
    model_dict = torch.load(args.load_from_checkpoint)["state_dict"]
    model_dict = {key.split("captioner.")[1]: value for key, value in model_dict.items() if "captioner." in key}
    model.load_state_dict(model_dict)
    model.eval()

    return model
