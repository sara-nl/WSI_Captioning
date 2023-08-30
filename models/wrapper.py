import torch
from torch import Tensor
import torch.nn as nn
import pytorch_lightning as pl

from .captioner import WSICaptioner, CLIP
from transformers import AutoModel

from .utils import *

from transformers import get_linear_schedule_with_warmup, BioGptForCausalLM
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

        # Load CWIP model in parts
        state_dict = torch.load(args.clip_checkpoint, map_location=torch.device("cpu"))["state_dict"]
        state_dict_visual = {key.split("model._orig_mod.visual.")[1]: value for key, value in state_dict.items() if "visual" in key}
        
        self.model.visual.load_state_dict(state_dict_visual)

        state_dict_text_projection = {key.split("model._orig_mod.text_projection.")[1]: value for key, value in state_dict.items() if "text_projection" in key}
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
        
        self.clip = CLIPLightning(capt_config, args)
        self.captioner = WSICaptioner(**capt_config)

        # make sure we don't train the pre-trained models
        self.clip.freeze()
        self.clip.eval()
 
        self.hyper_params = args
        
        print("Pre-trained MODELS LOADED!")

        # LM loss
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, text_hiddens: Tensor, visual_embeddings: Tensor):

        visual_embeddings = self.clip.model.visual(visual_embeddings)

        return self.captioner(text_hiddens, visual_embeddings)

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
        _, visual_embeddings, embedded_sequence, next_sequence = batch
        logits = self(embedded_sequence, visual_embeddings)

        loss = self.criterion(logits.permute(0,2,1), next_sequence)

        return {"loss": loss, "batch": batch}

    def training_step(self, batch: tuple, idx: int):
        outputs = self._step(batch, idx)
        self.log('train_loss', outputs["loss"], on_step=True)

        return outputs["loss"]  

    def validation_step(self, batch: tuple, idx: int):
        outputs = self._step(batch, idx)
        self.log('valid_loss', outputs["loss"], on_step=True)

        #TODO: this generated every step, preferably do this at the end
        key, visual_embeddings, _, token_sequence = batch 

        return {"key": key, "loss": outputs["loss"], 
                "visual_embeddings": visual_embeddings, 
                "tokenized_sequences": token_sequence}   
    
    def configure_optimizers(self):

        params = [
        {"params": self.captioner.parameters(), "lr": self.hyper_params.lm_lr}
        ]
        optimizer = torch.optim.AdamW(params, weight_decay=0.2, betas=(0.9,0.999),eps=1e-8)

        #lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.99)

        lr_scheduler = get_linear_schedule_with_warmup(
                        optimizer, num_warmup_steps=15, num_training_steps=120
                    )

        return {'optimizer': optimizer, 'lr_scheduler': lr_scheduler}
