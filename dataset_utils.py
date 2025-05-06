import torch
from torch.utils.data import Dataset
import numpy as np

class BehaviorDataset(Dataset):
    def __init__(self, behavior_df, embedding_df=None, use_h0=False):
        self.use_h0 = use_h0
        self.seqs = []
        self.targets = []
        self.h0_vecs = []
        self.block_ids = []  


        if self.use_h0:
            if embedding_df is None:
                raise ValueError("embedding_df must be provided if use_h0=True")
            df = behavior_df.merge(embedding_df, on=['subject_id', 'block_number'])
        else:
            df = behavior_df.copy()

       
        grouped = df.groupby(['subject_id', 'block_number'], sort=False)
        for (subj, block), group in grouped:
            group = group.copy()

      
            group['trial_index'] = range(len(group))

            group = group.sort_values('trial_index')
            group['prev_choice'] = group['bandit_chosen'].shift(1)
            group['prev_reward'] = group['reward_outcome'].shift(1)
            group = group.dropna(subset=['prev_choice', 'prev_reward'])

            if len(group) < 1:
                continue

            inputs = group[['prev_choice', 'prev_reward']].values.astype(np.float32)
            target = int(group['bandit_chosen'].iloc[-1])

            self.seqs.append(torch.tensor(inputs, dtype=torch.float32))
            self.targets.append(torch.tensor(target, dtype=torch.long))
            self.block_ids.append(f"{subj}_{block}")  
            if self.use_h0:
                emb = group['embedding'].iloc[0]
                if isinstance(emb, str):
                    emb = np.array(eval(emb))  
                self.h0_vecs.append(torch.tensor(emb, dtype=torch.float32))

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        if self.use_h0:
            return self.seqs[idx], self.h0_vecs[idx], self.targets[idx]
        else:
            return self.seqs[idx], self.targets[idx]
