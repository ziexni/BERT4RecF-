"""
datamodule.py
BERT4RecF+ DataModule with Multimodal Features
"""

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
from data import MicroVideoDataset, get_data, load_item_features

INTERACTION_PATH = 'datasets/interaction.parquet'
ITEM_PATH = 'datasets/item_used.parquet'
TITLE_NPY_PATH = 'datasets/title_emb.npy'


class DataModule(pl.LightningDataModule):
    def __init__(self, args):
        super(DataModule, self).__init__()

        # Hyperparameters
        self.max_len = args.max_len
        self.mask_prob = args.mask_prob
        self.neg_sample_size = args.neg_sample_size
        self.pin_memory = args.pin_memory
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.interaction_path = args.interaction_path

        # Multimodal
        self.use_multimodal = args.use_multimodal
        self.item_path = args.item_path
        self.title_npy_path = args.title_npy_path

        # Load data
        self.user_train, self.user_valid, self.user_test, self.usernum, self.itemnum = get_data(
            self.interaction_path
        )
        
        # Set item_size
        args.item_size = self.itemnum

        # Load multimodal features
        if self.use_multimodal:
            print("[DataModule] Loading multimodal features...")
            (self.text_feat, self.image_feat, self.category_feat,
             text_dim, image_dim, category_dim) = load_item_features(
                self.interaction_path,
                self.item_path,
                self.title_npy_path
            )
            args.text_dim = text_dim
            args.image_dim = image_dim
            args.category_dim = category_dim
            print(f"[DataModule] Multimodal loaded: text={text_dim}, "
                  f"image={image_dim}, category={category_dim}")
        else:
            self.text_feat = None
            self.image_feat = None
            self.category_feat = None
            args.text_dim = 0
            args.image_dim = 0
            args.category_dim = 0

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            # Train dataset
            self.train_dataset = MicroVideoDataset(
                self.user_train, self.user_valid, self.user_test,
                self.itemnum, self.max_len, self.mask_prob, mode='train',
                text_feat=self.text_feat,
                image_feat=self.image_feat,
                category_feat=self.category_feat
            )

            # Validation dataset
            self.valid_dataset = MicroVideoDataset(
                self.user_train, self.user_valid, self.user_test,
                self.itemnum, self.max_len,
                neg_sample_size=self.neg_sample_size,
                mode='valid',
                usernum=self.usernum,
                text_feat=self.text_feat,
                image_feat=self.image_feat,
                category_feat=self.category_feat
            )
        
        if stage == 'test' or stage is None:
            # Test dataset
            self.test_dataset = MicroVideoDataset(
                self.user_train, self.user_valid, self.user_test,
                self.itemnum, self.max_len,
                neg_sample_size=self.neg_sample_size,
                mode='test',
                usernum=self.usernum,
                text_feat=self.text_feat,
                image_feat=self.image_feat,
                category_feat=self.category_feat
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers
        )
    
    @staticmethod    
    def add_to_argparse(parser):
        parser.add_argument('--max_len', type=int, default=50)
        parser.add_argument('--mask_prob', type=float, default=0.2)
        parser.add_argument('--neg_sample_size', type=int, default=100)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--pin_memory', type=bool, default=True)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--item_size', type=int, default=0)
        parser.add_argument('--interaction_path', type=str, default=INTERACTION_PATH)
        
        # Multimodal
        parser.add_argument('--use_multimodal', action='store_true', default=False,
                            help='Enable multimodal features (text, image, category)')
        parser.add_argument('--item_path', type=str, default=ITEM_PATH)
        parser.add_argument('--title_npy_path', type=str, default=TITLE_NPY_PATH)
        parser.add_argument('--text_dim', type=int, default=0)
        parser.add_argument('--image_dim', type=int, default=0)
        parser.add_argument('--category_dim', type=int, default=0)

        return parser
