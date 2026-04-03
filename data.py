"""
data.py
BERT4RecF+ - Multimodal features 추가
"""

import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from collections import defaultdict


def load_item_features(interaction_path, item_path, title_npy_path):
    """
    멀티모달 피처 로딩 (SASRec과 동일 방식)
    
    Returns:
        text_feat: (max_item_id + 2, text_dim) - BERT는 MASK 토큰 때문에 +2
        image_feat: (max_item_id + 2, image_dim)
        category_feat: (max_item_id + 2, category_dim)
        text_dim, image_dim, category_dim
    """
    item_df = pd.read_parquet(item_path)
    title_raw = np.load(title_npy_path)  # (num_items, title_dim)

    # 1-based indexing (0=PAD, item_size+1=MASK)
    item_df = item_df.copy().reset_index(drop=True)
    item_df['item_id'] = item_df['item_id'] + 1
    max_item_id = int(item_df['item_id'].max())

    # ── Text (title_emb.npy) ──────────────────────────────────────────
    title_dim = title_raw.shape[1]
    text_feat = np.zeros((max_item_id + 2, title_dim), dtype=np.float32)  # +2 for MASK
    for raw_idx, row in item_df.iterrows():
        iid = int(row['item_id'])
        if raw_idx < len(title_raw):
            text_feat[iid] = title_raw[raw_idx].astype(np.float32)

    # ── Image/Video (video_feature) ───────────────────────────────────
    sample_img = item_df['video_feature'].iloc[0]
    image_dim = len(sample_img)
    image_feat = np.zeros((max_item_id + 2, image_dim), dtype=np.float32)
    for _, row in item_df.iterrows():
        iid = int(row['item_id'])
        image_feat[iid] = np.array(row['video_feature'], dtype=np.float32)

    # ── Category (multi-hot) ──────────────────────────────────────────
    all_cats = set()
    for cats in item_df['category_id']:
        if isinstance(cats, (list, np.ndarray)):
            all_cats.update(int(c) for c in cats)
        else:
            all_cats.add(int(cats))
    num_cats = max(all_cats) + 1

    category_feat = np.zeros((max_item_id + 2, num_cats), dtype=np.float32)
    for _, row in item_df.iterrows():
        iid = int(row['item_id'])
        cats = row['category_id']
        if isinstance(cats, (list, np.ndarray)):
            for c in cats:
                category_feat[iid, int(c)] = 1.0
        else:
            category_feat[iid, int(cats)] = 1.0

    print(f"[load_item_features] title_dim={title_dim}, image_dim={image_dim}, "
          f"category_dim={num_cats}, max_item_id={max_item_id}")

    return text_feat, image_feat, category_feat, title_dim, image_dim, num_cats


def get_data(interaction_path):
    """
    Interaction file load
    Returns:
        user_train, user_valid, user_test: {user: [item_ids]}
        usernum, itemnum
    """
    df = pd.read_parquet(interaction_path)
    df['user_id'] = df['user_id'] + 1
    df['item_id'] = df['item_id'] + 1
    df = df.sort_values(by=['user_id', 'timestamp'], kind='mergesort').reset_index(drop=True)

    usernum = df['user_id'].max()
    itemnum = df['item_id'].max()

    User = defaultdict(list)
    for u, i in zip(df['user_id'], df['item_id']):
        User[u].append(int(i))

    user_train, user_valid, user_test = {}, {}, {}
    for user, seq in User.items():
        n = len(seq)
        if n < 3:
            user_train[user] = seq
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = seq[:-2]
            user_valid[user] = [seq[-2]]
            user_test[user] = [seq[-1]]

    print(f"[Split] train users: {len(user_train)}, "
          f"valid: {sum(1 for v in user_valid.values() if v)}, "
          f"test: {sum(1 for v in user_test.values() if v)}")

    return user_train, user_valid, user_test, usernum, itemnum


class MicroVideoDataset(Dataset):
    """
    BERT4RecF+ Dataset with Multimodal Features
    
    Token 규칙:
        PAD = 0
        item = 1 ~ itemnum
        MASK = itemnum + 1
    """
    def __init__(self, user_train, user_valid, user_test,
                 itemnum, maxlen, mask_prob=0.2, neg_sample_size=100, mode='train',
                 usernum=0,
                 # Multimodal features
                 text_feat=None, image_feat=None, category_feat=None):
        
        self.user_train = user_train
        self.user_valid = user_valid
        self.user_test = user_test

        self.itemnum = itemnum
        self.maxlen = maxlen
        self.mask_prob = mask_prob
        self.neg_sample_size = neg_sample_size
        self.mask_token = itemnum + 1
        self.mode = mode
        self.item_size = itemnum

        # Multimodal features
        self.text_feat = text_feat
        self.image_feat = image_feat
        self.category_feat = category_feat
        self.use_multimodal = (text_feat is not None)

        # User list
        if mode == 'train':
            self.users = [u for u, seq in user_train.items() if len(seq) >= 2]
            if usernum > 10000:
                self.users = random.sample(self.users, min(10000, len(self.users)))
        else:
            ref = user_valid if mode == 'valid' else user_test
            self.users = [u for u in user_train if ref.get(u)]
            if usernum > 10000:
                self.users = random.sample(self.users, min(10000, len(self.users)))

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):
        u = self.users[idx]

        if self.mode == 'train':
            return self._train_item(u)
        else:
            return self._eval_item(u)

    def _train_item(self, u):
        """Train mode - random masking"""
        seq = self.user_train[u]

        tokens, labels = [], []

        for item in seq:
            if random.random() < self.mask_prob:
                tokens.append(self.mask_token)
                labels.append(item)
            else:
                tokens.append(item)
                labels.append(0)

        # Truncate
        tokens = tokens[-self.maxlen:]
        labels = labels[-self.maxlen:]

        # Padding
        pad_len = self.maxlen - len(tokens)
        tokens = [0] * pad_len + tokens
        labels = [0] * pad_len + labels

        tokens_t = torch.LongTensor(tokens)
        labels_t = torch.LongTensor(labels)

        if self.use_multimodal:
            # Extract multimodal features
            text_t = torch.FloatTensor(self.text_feat[tokens])
            image_t = torch.FloatTensor(self.image_feat[tokens])
            category_t = torch.FloatTensor(self.category_feat[tokens])
            return tokens_t, labels_t, text_t, image_t, category_t
        
        return tokens_t, labels_t

    def _eval_item(self, u):
        """Validation/Test mode - predict next item"""
        train_seq = self.user_train.get(u, [])

        if self.mode == 'valid':
            history = train_seq
            target = self.user_valid[u][0]
        else:
            history = self.user_train[u] + self.user_valid.get(u, [])
            target = self.user_test[u][0]

        # Input sequence
        item_seq = [iid for iid in history]
        item_seq = item_seq[-(self.maxlen - 1):]
        seq = item_seq + [self.mask_token]

        # Padding
        pad_len = self.maxlen - len(seq)
        seq = [0] * pad_len + seq

        # Negative sampling
        rated = set(self.user_train[u])
        rated.add(0)
        negs = []

        while len(negs) < self.neg_sample_size:
            t = np.random.randint(1, self.itemnum + 1)
            if t not in rated:
                negs.append(t)

        # Candidates (101 = 1 positive + 100 negative)
        candidates = [target] + negs
        labels = [1] + [0] * self.neg_sample_size

        seq_t = torch.LongTensor(seq)
        candidates_t = torch.LongTensor(candidates)
        labels_t = torch.LongTensor(labels)

        if self.use_multimodal:
            # Sequence features
            text_seq_t = torch.FloatTensor(self.text_feat[seq])
            image_seq_t = torch.FloatTensor(self.image_feat[seq])
            category_seq_t = torch.FloatTensor(self.category_feat[seq])

            # Candidate features
            text_cand_t = torch.FloatTensor(self.text_feat[candidates])
            image_cand_t = torch.FloatTensor(self.image_feat[candidates])
            category_cand_t = torch.FloatTensor(self.category_feat[candidates])

            return (seq_t, candidates_t, labels_t,
                    text_seq_t, image_seq_t, category_seq_t,
                    text_cand_t, image_cand_t, category_cand_t)

        return seq_t, candidates_t, labels_t
