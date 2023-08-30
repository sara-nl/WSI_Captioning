import yaml
from pytorch_lightning import Trainer
from data.image_text_loader import TextImageDataModule
from models import CaptioningWrapper
from pytorch_lightning.callbacks import LearningRateMonitor
from models.utils import get_parser

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
    
    args = get_parser(Trainer)

    main(args)
