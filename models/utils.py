import os
import torch
from pathlib import Path
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple
from .captioner import WSICaptioner, CLIP

import numpy as np
import pandas as pd

from argparse import ArgumentParser

from transformers import BioGptForCausalLM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    state_dict_visual = {key.split("model.visual.")[1]: value for key, value in state_dict.items() if "model.visual" in key}
    clip.visual.load_state_dict(state_dict_visual)

    state_dict_text_projection = {key.split("model.text_projection.")[1]: value for key, value in state_dict.items() if "model.text_projection" in key}
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

def get_val_data(args):
    validation_folds = [args.val_fold]

    lmdb_patches = os.listdir(args.lmdb_patches_path)
    lmdb_patches = {lmdb_patch.split(".pt")[0]: lmdb_patch for lmdb_patch in lmdb_patches}

    listed_data_df = pd.read_csv(args.crossvalidation_path)
    
    validation_folds = listed_data_df[listed_data_df["Fold"].isin(validation_folds)]

    if args.dataset=="radboud":
        validation_folds = validation_folds[validation_folds["WSI"].str.contains("EX")]

    elif args.dataset=="catania":
        validation_folds = validation_folds[~validation_folds["WSI"].str.contains("EX")]

    listed_val_data = validation_folds.WSI.tolist()
    listed_val_data = [lmdb_patches[wsi] for wsi in listed_val_data if wsi in lmdb_patches]
    listed_val_labels = [wsi_labels[1:-2].astype(int) for wsi_labels in validation_folds.to_numpy() if wsi_labels[0] in lmdb_patches]

    return listed_val_data, listed_val_labels

def get_diagnostic_report(args, key):
    text_path = os.path.join(args.texts_path, key.split(".pt")[0]+".txt")
    if Path(text_path).exists():
        with open(text_path, "r") as f:
                text = f.read()

    else:
        text_path = os.path.join(args.texts_path, key.split(".")[0]+".mrxs.txt")
        if Path(text_path).exists():
            with open(text_path, "r") as f:
                text = f.read()

    return text

def get_sentence_embeddings(args, model, text=None, tokenizer=None, input_tokens=None, output_all_hiddens: bool = False):

    if tokenizer is not None:
   
        input_tokens = tokenizer.batch_encode_plus(text, return_tensors='pt', 
                                                padding="max_length", 
                                                max_length=args.context_length+1, 
                                                truncation=True).to(device) 
        
        input_tokens = input_tokens["input_ids"]

    outputs = model(input_ids=input_tokens, output_hidden_states=True)
    hidden_states = outputs.hidden_states[-2]

    if output_all_hiddens:
        return hidden_states

    sentence_embedding = hidden_states[:,0]
    return sentence_embedding    

@torch.no_grad()
def autoregress(idx: Tensor, 
                args: ArgumentParser,
                model: WSICaptioner,
                visual_embedding: Tensor, 
                max_new_tokens: int, 
                encoder_model: BioGptForCausalLM = None,
                temperature: float = 1.0, 
                do_sample: bool = False, 
                top_k: Tensor = None,
                repetition_penalty: bool = False) -> Tensor:
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    """

    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= args.context_length else idx[:, -args.context_length:]

        hiddens = get_sentence_embeddings(args, encoder_model, input_tokens=idx_cond, output_all_hiddens=True)
        #hiddens = idx_cond
        logits = model(hiddens, visual_embedding)
        # pluck the logits at the final step and scale by desired temperature
        logits = logits[:, -1, :] / temperature
        # optionally crop the logits to only the top k options
        if top_k is not None:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = -float('Inf')
        # apply softmax to convert logits to (normalized) probabilities
        probs = F.softmax(logits, dim=-1)

        # UNTESTED FOR NOW
        if repetition_penalty:
            # Typically 0.8 but we can make it an optional
            penalty = 0.8
            score = torch.gather(probs, 1, idx)

            # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
            score = torch.where(score < 0, score * penalty, score / penalty)
            probs.scatter_(1, idx, score)

        # either sample from the distribution or take the most likely element
        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)

        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)

        # append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)

    return idx
    
def generate(args, 
             prompt: str, 
             model: WSICaptioner,
             visual_embedding: Tensor, 
             encoder_model: BioGptForCausalLM = None,
             num_samples: int = 1, 
             do_sample: bool = False, 
             tokenizer=None):
        """
        TODO: Greedy encoding for now, test for beam search and expand this
        function to output to files/do more elaborate stuff.

        inspired by https://huggingface.co/blog/how-to-generate
        """

        # encode the starting prompt
        x = tokenizer.encode(prompt, return_tensors='pt').to(device)
        # Take only the first token as input
        x = x[:,0]       
                
        # we'll process all desired num_samples in a batch, so expand out the batch dim
        x = x.expand(num_samples, -1)
        visual_embedding = visual_embedding.expand(num_samples, -1)

        # forward the model context_length-1 times times to get samples, in a batch
        generated_tokens = autoregress(x, 
                                       args,
                                       model, 
                                       visual_embedding, 
                                       args.context_length-1,
                                       encoder_model=encoder_model,
                                       do_sample=do_sample,
                                       top_k=None, 
                                       temperature=args.temperature)     
        
        #print("generated tokens: ", generated_tokens)
        generated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return generated, generated_tokens
