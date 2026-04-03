import argparse
import datetime
from datamodule import DataModule
from lit_model import BERT4RecFPlus
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint


def _setup_parser():
    """
    CLI argument parser
    """
    parser = argparse.ArgumentParser(description='BERT4RecF+ - MicroVideo with Multimodal')
    
    # Data arguments
    data_group = parser.add_argument_group("Data Args")
    DataModule.add_to_argparse(data_group)
    
    # Model arguments
    model_group = parser.add_argument_group("Model Args")
    BERT4RecFPlus.add_to_argparse(model_group)
    
    return parser


def _set_trainer_args(args):
    """
    Trainer configuration
    """
    args.max_epochs = 100
    args.gradient_clip_val = 5.0
    args.gradient_clip_algorithm = "norm"
    
    # Logging
    model_name = "BERT4RecF+" if args.use_multimodal else "BERT4Rec"
    args.save_dir = "Training/logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.name = f"{model_name}_MicroVideo"
    
    # Early stopping & Checkpoint - NDCG 기준 (다른 베이스라인과 동일)
    args.monitor = "NDCG_val"
    args.mode = "max"
    args.patience = 5
    
    # Logging interval
    args.logging_interval = "step"


def main():
    """
    Main training pipeline
    """
    # Parse arguments
    parser = _setup_parser()
    args = parser.parse_args()
    
    # Trainer config
    _set_trainer_args(args)
    
    print("="*50)
    print(f"Model: {'BERT4RecF+ (Multimodal)' if args.use_multimodal else 'BERT4Rec (Baseline)'}")
    print("="*50)
    
    # DataModule
    data = DataModule(args)
    
    # Model
    lit_model = BERT4RecFPlus(args)
    
    # Logger
    logger = TensorBoardLogger(
        save_dir=args.save_dir,
        name=args.name
    )
    
    # Callbacks
    early_stop = EarlyStopping(
        monitor=args.monitor,
        mode=args.mode,
        patience=args.patience
    )
    
    lr_monitor = LearningRateMonitor(
        logging_interval=args.logging_interval
    )
    
    checkpoint = ModelCheckpoint(
        monitor='NDCG_val',
        mode='max',
        save_top_k=1,
        filename='best'
    )
    
    # Trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        gradient_clip_val=args.gradient_clip_val,
        gradient_clip_algorithm=args.gradient_clip_algorithm,
        check_val_every_n_epoch=1,
        logger=logger,
        callbacks=[early_stop, lr_monitor, checkpoint],
        accelerator='gpu',
        devices=1,
    )
    
    # Training
    print("\n" + "="*50)
    print("Starting Training...")
    print("="*50 + "\n")
    
    trainer.fit(lit_model, datamodule=data)
    
    # Testing
    print("\n" + "="*50)
    print("Testing...")
    print("="*50 + "\n")
    
    trainer.test(lit_model, datamodule=data, ckpt_path='best')
    
    print("\n" + "="*50)
    print("Done!")
    print("="*50)


if __name__ == "__main__":
    main()
