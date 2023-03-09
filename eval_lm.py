import os
import torch
from pathlib import Path
from torch import Tensor
import torch.nn.functional as F
import yaml
from models import WSICaptioner, CLIP

from sacrebleu.metrics import BLEU
from rouge import Rouge

import numpy as np
import pandas as pd

from transformers import BioGptTokenizer, BioGptForCausalLM, AutoModel, AutoTokenizer
from argparse import ArgumentParser

from models.utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):

    bio_gpt = BioGptForCausalLM.from_pretrained("microsoft/biogpt").to("cuda:0")
    print("BIO-GPT Loaded!")
    tokenizer_biogpt = BioGptTokenizer.from_pretrained("microsoft/biogpt")

    pubmedbert = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    print("PubMedBERT Loaded!")
    
    pubmedbert = pubmedbert.to("cuda:0")
    tokenizer_pubmed = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

    with open(args.captioning_config_dir) as fin:
        capt_config = yaml.safe_load(fin)["Capt"]

    clip = load_CLIP(capt_config, args)
    print("CLIP Loaded!")

    model = load_LM_model(capt_config, args)
    print("Captioning Model Loaded!")

    print("MODELS ARE LOCKED and LOADED!")

    listed_val_data, listed_val_labels = get_val_data(args)
    N = len(listed_val_data)

    print("DATA HAS BEEN LOADED!")
    bleu = BLEU()
    rouge = Rouge()
    
    batch_size = 1 if args.rank else args.generate_batch_size
        
    bleu_scores = np.zeros((N, batch_size))
    rouge_scores= np.zeros((N, batch_size))
    text_similarities = np.zeros((N, batch_size))
    image_similarities = np.zeros((N, batch_size))

    with torch.no_grad():
        for wsi_number in range(N):
            print(wsi_number)
            # Get WSI embeddings
            key = listed_val_data[wsi_number]
            labels = listed_val_labels[wsi_number]
            #print(labels)
            
            visual_embeddings = torch.load(str(Path(args.lmdb_patches_path) / f"{Path(key)}"))
            visual_embeddings = visual_embeddings.mean(dim=0).unsqueeze(0).to(device)
            clip_visual_embeddings = clip.visual(visual_embeddings)
            
            # Get text embedding
            text = get_diagnostic_report(args, key)
            real_text_embedding = get_sentence_embeddings(args, pubmedbert, text=[text], tokenizer=tokenizer_pubmed)                      
            print("real text: ", text)
            # Generate candidates
            generated, _ = generate(args, 
                                    "",
                                    model, 
                                    visual_embeddings,
                                    encoder_model=bio_gpt,
                                    num_samples=args.generate_batch_size,
                                    do_sample=args.sample, tokenizer=tokenizer_biogpt)

            # convert the generations to pubmed embeddings so we can use CLIP
            generated_embedding = get_sentence_embeddings(args, pubmedbert, text=generated, tokenizer=tokenizer_pubmed)

            # Compute cosine similarities generated_text <-> real_text and generated_text <-> image_embeddings
            clip_generated_embedding = F.normalize(clip.text_projection(generated_embedding), dim=1)
            clip_report_embedding = F.normalize(clip.text_projection(real_text_embedding), dim=1)

            clip_visual_embeddings = F.normalize(clip_visual_embeddings, dim=1)
            
            # Rank and choose best generation according to CLIP
            if args.rank:
                # No need to rank reports w.r.t. generations
                report_sorted, report_indices = ranked_cosine_similarity(clip_report_embedding, clip_generated_embedding)
                visual_sorted, visual_indices = ranked_cosine_similarity(clip_visual_embeddings, clip_generated_embedding)

                # closest image candidate
                generated = [generated[visual_indices[0,0]]]

                image_similarity = visual_sorted[0,0].item()
                report_similarity = report_sorted[0,0].item()

            else:
                report_similarity = clip_report_embedding @ clip_generated_embedding.t()    
                image_similarity = clip_visual_embeddings @ clip_generated_embedding.t()
                report_similarity = report_similarity.cpu().numpy()
                image_similarity = image_similarity.cpu().numpy()   
                
            # Compute metrics: bleu, rouge and similarities
            print("generated text: ", generated[0])
            bleu_scores[wsi_number,:] = [bleu.corpus_score([text], gen).score / 100 for gen in generated]
            rouge_scores[wsi_number,:] = [rouge.get_scores(hyps=gen, refs=text)[0]["rouge-l"]["f"] for gen in generated] 

            text_similarities[wsi_number,:] = report_similarity
            image_similarities[wsi_number,:] = image_similarity
       
    print("bleu mean: ", bleu_scores.mean())
    print("rouge mean: ", rouge_scores.mean())
    print("text sim mean: ", text_similarities.mean())
    print("image sim mean: ", image_similarities.mean())

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--load_from_checkpoint', default="./checkpoints/CAPT/", type=str, help='choose captioning model to be loaded')
    parser.add_argument('--captioning_config_dir', type=str, default='models/configs/Capt.yaml')
    parser.add_argument('--dataset', default="both", type=str, help='choose between radboud, catania or both')
    parser.add_argument('--crossvalidation_path', type=str, required=True, help='path to the .csv data containing the partitions')
    parser.add_argument('--clip_model_path', type=str, required=True, help='path to the pre-trained CLIP model')

    parser.add_argument('--lmdb_patches_path', type=str, required=True, help='directory of your training folder')
    parser.add_argument('--texts_path', type=str, required=True, help='directory of your texts')

    parser.add_argument('--generate_batch_size', type=int, default=32, help='size of the batch')
    parser.add_argument('--context_length', type=int, default=32, help='context')
    parser.add_argument('--temperature', type=float, default=0.95, help='gen temp')
    parser.add_argument('--val_fold', type=int, default=9, required=True)
    parser.add_argument('--sample', action='store_true')
    parser.add_argument('--rank', action='store_true')

    args = parser.parse_args()

    main(args)