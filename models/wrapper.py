import time
import numpy as np
import json

import torch
from torch import Tensor
from typing import Tuple, Optional
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .captioner import WSICaptioner, CLIP
from transformers import AutoModel

from transformers import AutoTokenizer, AutoTokenizer, get_linear_schedule_with_warmup, BioGptTokenizer, BioGptForCausalLM
from argparse import ArgumentParser

class BioGPT(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")

    def forward(self, input_ids: Tensor, output_hidden_states: bool=True):
        return self.model(input_ids=input_ids, output_hidden_states=output_hidden_states)

class PubMedBERT(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model = AutoModel.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
        
    def forward(self, input_ids: Tensor, output_hidden_states: bool=True):
        return self.model(input_ids=input_ids, output_hidden_states=output_hidden_states)

class CLIPLightning(pl.LightningModule):
    def __init__(self, capt_config: dict, args: ArgumentParser):
        super().__init__()
        self.model = CLIP(embed_dim=capt_config["embed_dim"], context_length=capt_config["context_length"])

        # Load CLIP model in parts
        state_dict = torch.load(args.clip_checkpoint, map_location=torch.device("cpu"))["state_dict"]

        state_dict_visual = {key.split("model.visual.")[1]: value for key, value in state_dict.items() if "visual" in key}
        self.model.visual.load_state_dict(state_dict_visual)

        state_dict_text_projection = {key.split("model.text_projection.")[1]: value for key, value in state_dict.items() if "text_projection" in key}
        self.model.text_projection.load_state_dict(state_dict_text_projection)

class CaptioningWrapper(pl.LightningModule):
    def __init__(self,
                 capt_config: dict,
                 args: ArgumentParser
                 ):
        """A lightning wrapper for a WSI Captiong model as specified in the google docs.

        Args:
        """
        super().__init__()
        # In this way the biogpt model gets loaded and saved when training
        # maybe better to extract the latents beforehand and load ths in validation_end_epoch?
        # Idem for pubmedbert
        self.biogpt = BioGPT()
        self.pubmedbert = PubMedBERT()
        self.clip = CLIPLightning(capt_config, args)
        self.captioner = WSICaptioner(**capt_config)

        # make sure we don't train the pre-trained models
        self.biogpt.freeze()
        self.pubmedbert.freeze()
        self.clip.freeze()

        self.biogpt.eval()
        self.pubmedbert.eval()
        self.clip.eval()

        # tokenizers
        self.biogpt_tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
        self.pubmed_tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
 
        self.hyper_params = args
        
        print("Pre-trained MODELS LOADED!")

        # LM loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, tokens: Tensor, visual_embeddings: Tensor):

        visual_clip_embeddings = self.clip.model.visual(visual_embeddings)
        hiddens = self._get_sentence_embeddings(self.biogpt, tokens, output_all_hiddens=True)

        return self.captioner(hiddens, visual_clip_embeddings)

    def _get_sentence_embeddings(self, model, input_tokens: Tensor, output_all_hiddens: bool = False) -> Tensor:
        """
        Get the embeddings from a model for a piece of text. 

        This can either be a sentence embedding from the [CLS] token in BERT or
        the embeddings for every token in the sequence from a GPT-style LM. 
        """

        outputs = model(input_ids=input_tokens, output_hidden_states=True)

        hidden_states = outputs.hidden_states[-2]

        # return all hiddens (N,L,H), not just the sentence embedding 
        if output_all_hiddens:
            return hidden_states

        # return the sentence embedding (N,1,H)
        sentence_embedding = hidden_states[:,0]
        return sentence_embedding    

    def _step(self, batch: tuple, batch_idx: int) -> dict:
        _, visual_embeddings, tokenized_sequence, next_sequence = batch
           
        logits = self(tokenized_sequence, visual_embeddings)
        logits = logits.permute(0,2,1)

        loss = self.criterion(logits, next_sequence)

        return {"loss": loss, "batch": batch}

    def training_step(self, batch: tuple, idx: int):
        outputs = self._step(batch, idx)
        self.log('train_loss', outputs["loss"], on_step=True)

        return outputs["loss"]  

    def validation_step(self, batch: tuple, idx: int):
        outputs = self._step(batch, idx)
        self.log('valid_loss', outputs["loss"], on_step=True)

        #TODO: this generated every step, preferably do this at the end
        key, visual_embeddings, tokenized_sequences, _ = batch 

        return {"key": key, "loss": outputs["loss"], 
                "visual_embeddings": visual_embeddings, 
                "tokenized_sequences": tokenized_sequences}   
   
    def _ranked_cosine_similarity(self, tensor_1: Tensor, tensor_2: Tensor) -> Tuple[Tensor]:
        """
        Compute cosine similarity between two tensors and rank it according to the similarity.
        """
        dot_similarity = tensor_1 @ tensor_2.t()
        sorted, indices = torch.topk(dot_similarity, k=dot_similarity.shape[1], dim=1)
        indices = indices.cpu().numpy()
        text_std = torch.std(sorted)

        # Get the best cosine similarities in the batch 
        best_similarity = sorted[0,:].cpu().numpy().mean()

        return best_similarity, indices, text_std
    
    def validation_epoch_end(self, val_step_outputs: dict):
        """
        Bulky function to generate for each WSI<->Report in the validation set
        a set of candidate generations (NUM_SAMPLES_VAL). We choose the best candidate
        and compute cosine similarities to the real text and to the WSI embedding.

        Slow function because of autoregression.
        """

        start_eval = time.time()
        visual_embeddings = torch.cat([batch["visual_embeddings"] for batch in val_step_outputs], dim=0)
        clip_visual_embeddings = self.clip.model.visual(visual_embeddings)

        tokenized_sequences = torch.cat([batch["tokenized_sequences"] for batch in val_step_outputs], dim=0)
        keys = [key for batch in val_step_outputs for key in batch["key"]]
        
        image_proximities = []
        text_proximities = []
        image_stds = []
        text_stds = []
        generated_texts = []
        real_texts = []

        for i, key in enumerate(keys):

            visual_embedding = visual_embeddings[i].unsqueeze(0)
            tokenized_sequence = tokenized_sequences[i].unsqueeze(0)
            # generate candidate diagnostic reports
            generated_texts_tokens, generated_tokens = self._generate("", 
                                                visual_embedding, 
                                                num_samples=self.hyper_params.num_samples_val, 
                                                do_sample=True, 
                                                tokenizer=self.biogpt_tokenizer)

            # get the input embeddings for the original and the generated texts
            real_text = [self.biogpt_tokenizer.decode(sent, skip_special_token=True) for sent in tokenized_sequence.detach().cpu().tolist()]
            
            tokenized_pubmed = self.pubmed_tokenizer.batch_encode_plus(real_text, 
                                                                       return_tensors='pt', 
                                                                       padding="max_length", 
                                                                       max_length=self.hyper_params.context_length+1, 
                                                                       truncation=True)["input_ids"].to(self.device)

            generated_tokens = self.pubmed_tokenizer.batch_encode_plus(generated_texts_tokens, 
                                                                       return_tensors='pt', 
                                                                       padding="max_length",
                                                                       max_length=self.hyper_params.context_length+1, 
                                                                       truncation=True)["input_ids"].to(self.device)

            
            sentence_embedding = self._get_sentence_embeddings(self.pubmedbert, tokenized_pubmed)
            generated_embedding = self._get_sentence_embeddings(self.pubmedbert, generated_tokens)

            # get the clip embeddings belonging to the texts
            clip_generated_embedding = F.normalize(self.clip.model.text_projection(generated_embedding), dim=1)
            clip_sentence_embedding = F.normalize(self.clip.model.text_projection(sentence_embedding), dim=1)

            best_text_proximity, indices, text_std = self._ranked_cosine_similarity(clip_sentence_embedding, clip_generated_embedding)
            
            #generated_text = generated_texts_tokens[indices[0,0]]

            real_text = real_text[0]

            clip_visual_embedding = F.normalize(clip_visual_embeddings[i].unsqueeze(0), dim=1)
            best_image_proximity, indices, image_std = self._ranked_cosine_similarity(clip_visual_embedding, clip_generated_embedding)

            # Get the generated text from the generated candidates that is most similar to the WSI
            generated_text = generated_texts_tokens[indices[0,0]]
            real_texts.append(real_text)
            generated_texts.append(generated_text)
           
            image_proximities.append(best_image_proximity)
            text_proximities.append(best_text_proximity)
            image_stds.append(image_std.cpu().numpy())
            text_stds.append(text_std.cpu().numpy())
            
        
        mean_img_prox = np.array(image_proximities).mean()
        mean_text_prox = np.array(text_proximities).mean()

        self.log('mean image proximity', mean_img_prox)
        self.log('mean text proximity', mean_text_prox)

        mean_img_std = np.array(image_stds).mean()
        mean_text_std = np.array(text_stds).mean()
        self.log('mean std image proximity', mean_img_std)
        self.log('mean std text proximity', mean_text_std)
        print("time: ", time.time() - start_eval)

        # output to file
        results_dict = {}
        if self.current_epoch==self.hyper_params.max_epochs-1:
            for i in range(len(generated_texts)):
                results_dict[i] = {"real_text: ": real_texts[i], 
                               "generated_text: ": generated_texts[i], 
                               "cosine_to_image": float(image_proximities[i])}

            with open(self.hyper_params.save_generations_path, "w") as file:
                json.dump(results_dict, file)
    
    @torch.no_grad()
    def _autoregress(self, 
                     idx: Tensor, 
                     visual_embedding: Tensor, 
                     max_new_tokens: int, 
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
            idx_cond = idx if idx.size(1) <= self.hyper_params.context_length else idx[:, -self.hyper_params.context_length:]
            logits = self(idx_cond, visual_embedding)
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
                # Typically 0.8 but can make it an optional
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
    
    def _generate(self, prompt: str, 
                  visual_embedding: Tensor, 
                  num_samples: int = 1, 
                  do_sample: bool = False, 
                  tokenizer=None):
        """
        TODO: Greedy encoding for now, test for beam search and expand this
        function to output to files/do more elaborate stuff.

        inspired by https://huggingface.co/blog/how-to-generate
        """

        # encode the starting prompt
        x = tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        # Take only the first token as input
        x = x[:,0]       
                
        # we'll process all desired num_samples in a batch, so expand out the batch dim
        x = x.expand(num_samples, -1)
        visual_embedding = visual_embedding.expand(num_samples, -1)

        # forward the model context_length-1 times times to get samples, in a batch
        generated_tokens = self._autoregress(x, visual_embedding, 
                                    self.hyper_params.context_length-1, 
                                    do_sample=do_sample,
                                    top_k=None, 
                                    temperature=self.hyper_params.temperature)     
                     
        generated = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

        return generated, generated_tokens

    def configure_optimizers(self):

        params = [
        {"params": self.captioner.parameters(), "lr": self.hyper_params.lm_lr}
        ]
        optimizer = torch.optim.AdamW(params, weight_decay=0.2, betas=(0.9,0.999),eps=1e-8)

        #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

        lr_scheduler = get_linear_schedule_with_warmup(
                        optimizer, num_warmup_steps=20, num_training_steps=130
                    )

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
