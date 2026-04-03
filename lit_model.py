import pytorch_lightning as pl
from torchmetrics import RetrievalHitRate, RetrievalNormalizedDCG, RetrievalMRR
import torch
import torch.nn as nn
import numpy as np

from bert import BERT


class BERT4RecFPlus(pl.LightningModule):
    """
    BERT4RecF+ - BERT4Rec with Multimodal Fusion
    """
    def __init__(self, args):
        super(BERT4RecFPlus, self).__init__()

        # Hyperparameters
        self.learning_rate = args.learning_rate
        self.max_len = args.max_len
        self.hidden_dim = args.hidden_dim
        self.encoder_num = args.encoder_num
        self.head_num = args.head_num
        self.dropout_rate = args.dropout_rate
        self.dropout_rate_attn = args.dropout_rate_attn

        # Vocab: 0=PAD, 1~item_size=items, item_size+1=MASK
        self.vocab_size = args.item_size + 2

        self.initializer_range = args.initializer_range
        self.weight_decay = args.weight_decay
        self.decay_step = args.decay_step
        self.gamma = args.gamma

        # Multimodal
        self.use_multimodal = args.use_multimodal

        # BERT encoder with multimodal fusion
        self.model = BERT(
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            hidden_dim=self.hidden_dim,
            encoder_num=self.encoder_num,
            head_num=self.head_num,
            dropout_rate=self.dropout_rate,
            dropout_rate_attn=self.dropout_rate_attn,
            initializer_range=self.initializer_range,
            # Multimodal
            text_dim=args.text_dim if self.use_multimodal else 0,
            image_dim=args.image_dim if self.use_multimodal else 0,
            category_dim=args.category_dim if self.use_multimodal else 0,
            use_multimodal=self.use_multimodal
        )

        # Output head
        self.out = nn.Linear(self.hidden_dim, args.item_size + 1)

        self.batch_size = args.batch_size

        # Loss (ignore_index=0 for PAD)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)

        # Metrics
        self.HR = RetrievalHitRate(top_k=10)
        self.NDCG = RetrievalNormalizedDCG(top_k=10)
        self.MRR = RetrievalMRR()

    def training_step(self, batch, batch_idx):
        """
        Training step with optional multimodal features
        """
        if self.use_multimodal:
            seq, labels, text_feat, image_feat, category_feat = batch
            logits = self.model(seq, text_feat=text_feat, 
                               image_feat=image_feat, 
                               category_feat=category_feat)
        else:
            seq, labels = batch
            logits = self.model(seq)

        preds = self.out(logits)  # (B, T, item_size + 1)

        # CrossEntropyLoss
        loss = self.criterion(preds.transpose(1, 2), labels)

        self.log("train_loss", loss,
                 on_step=True, on_epoch=True,
                 prog_bar=True, logger=True)
    
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step with ranking evaluation
        """
        if self.use_multimodal:
            (seq, candidates, labels,
             text_seq, image_seq, category_seq,
             text_cand, image_cand, category_cand) = batch
            
            # Encode sequence
            logits = self.model(seq, text_feat=text_seq,
                               image_feat=image_seq,
                               category_feat=category_seq)
        else:
            seq, candidates, labels = batch
            logits = self.model(seq)

        preds = self.out(logits)

        # Last position prediction
        preds = preds[:, -1, :]  # (B, item_size + 1)
        
        # Target item
        targets = candidates[:, 0]

        # Classification loss
        loss = self.criterion(preds, targets)

        # Candidate scores
        recs = torch.gather(preds, 1, candidates)

        # Indexes for metrics
        steps = batch_idx * self.batch_size
        indexes = torch.arange(
            steps, steps + seq.size(0),
            dtype=torch.long,
            device=seq.device
        ).unsqueeze(1).repeat(1, 101)

        # Logging
        self.log("val_loss", loss,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.log("HR_val",
                 self.HR(recs, labels, indexes),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.log("NDCG_val",
                 self.NDCG(recs, labels, indexes),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.log("MRR_val",
                 self.MRR(recs, labels, indexes),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def test_step(self, batch, batch_idx):
        """
        Test step (same as validation)
        """
        if self.use_multimodal:
            (seq, candidates, labels,
             text_seq, image_seq, category_seq,
             text_cand, image_cand, category_cand) = batch
            
            logits = self.model(seq, text_feat=text_seq,
                               image_feat=image_seq,
                               category_feat=category_seq)
        else:
            seq, candidates, labels = batch
            logits = self.model(seq)

        preds = self.out(logits)

        preds = preds[:, -1, :]
        targets = candidates[:, 0]
        loss = self.criterion(preds, targets)

        recs = torch.gather(preds, 1, candidates)

        steps = batch_idx * self.batch_size
        indexes = torch.arange(
            steps, steps + seq.size(0),
            dtype=torch.long,
            device=seq.device
        ).unsqueeze(1).repeat(1, 101)

        self.log("test_loss", loss,
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.log("HR_test",
                 self.HR(recs, labels, indexes),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.log("NDCG_test",
                 self.NDCG(recs, labels, indexes),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.log("MRR_test",
                 self.MRR(recs, labels, indexes),
                 on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Inference - top-10 item recommendations
        """
        if self.use_multimodal:
            seq, text_feat, image_feat, category_feat = batch
            logits = self.model(seq, text_feat=text_feat,
                               image_feat=image_feat,
                               category_feat=category_feat)
        else:
            seq = batch
            logits = self.model(seq)

        preds = self.out(logits)
        preds = preds[:, -1, :]  # Last position

        # Top-10
        indexes, _ = torch.topk(preds, 10)

        return indexes.cpu().numpy()
    
    def configure_optimizers(self):
        """
        Optimizer + scheduler
        """
        # Weight decay separation
        no_decay = ['bias', 'LayerNorm.weight']

        params = [
            {
                'params': [p for n, p in self.named_parameters()
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.weight_decay
            },
            {
                'params': [p for n, p in self.named_parameters()
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = torch.optim.Adam(params, lr=self.learning_rate)

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.decay_step,
            gamma=self.gamma
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "val_loss"
        }
    
    @staticmethod
    def add_to_argparse(parser):
        parser.add_argument("--learning_rate", type=float, default=1e-3)
        parser.add_argument("--hidden_dim", type=int, default=256)
        parser.add_argument("--encoder_num", type=int, default=2)
        parser.add_argument("--head_num", type=int, default=4)
        parser.add_argument("--dropout_rate", type=float, default=0.1)
        parser.add_argument("--dropout_rate_attn", type=float, default=0.1)
        parser.add_argument("--initializer_range", type=float, default=0.02)
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--decay_step", type=int, default=25)
        parser.add_argument("--gamma", type=float, default=0.1)
        return parser
