# datasets/beat2motion_dataset.py
import os, random, codecs as cs, numpy as np, torch
from os.path import join as pjoin
from torch.utils.data import Dataset
from tqdm import tqdm

# Optional helpers; stub them or replace by your own
# from datasets.emage_utils.motion_rep_transfer import get_motion_rep_numpy

class Beat2MotionDataset(Dataset):
    """
    Dataset for BEAT‑like motion‑text clips stored as
      ├─ npy_segments/xxx.npy          (1, T, 264)
      └─ txt_segments/xxx.txt          captions / tags
    Ready for diffusion‑based motion generation.
    """
    def __init__(
        self,
        opt,                       # Hyper‑parameters & paths holder (any object with the attrs below)
        mean: np.ndarray | None,   # motion mean vector  (D,)
        std:  np.ndarray | None,   # motion std  vector  (D,)
        split_file: str,           # txt: list of clip ids to load
        times: int = 1,            # dataset “augmentation” factor
        w_vectorizer=None,         # optional word ⇒ (embedding, POS‑one‑hot)
        eval_mode: bool = False,   # True → return embeddings instead of raw caption
    ):
        super().__init__()

        self.opt         = opt
        self.times       = times
        self.w_vectorizer = w_vectorizer
        self.eval_mode   = eval_mode
        self.fps         = 60                          # ← your dataset FPS
        self.min_len     = 40 if opt.dataset_name == 'beat' else 24
        self.max_len     = opt.max_motion_length
        self.use_rep     = opt.motion_rep              # 'axis_angle' | 'rep15d' | 'position'

        # -------------------------------------------------
        # 1. Read split file
        # -------------------------------------------------
        with cs.open(split_file, 'r', encoding='utf‑8') as f:
            clip_ids = [ln.strip() for ln in f if ln.strip()]

        self.data_dict, name_lens = {}, []
        for cid in tqdm(clip_ids, desc='Loading motion‑text pairs'):
            try:
                motion_path = pjoin(opt.motion_dir, f'{cid}.npy')
                txt_path    = pjoin(opt.text_dir,   f'{cid}.txt')
                if not (os.path.exists(motion_path) and os.path.exists(txt_path)):
                    continue

                motion_raw = np.load(motion_path)
                if motion_raw.ndim == 3 and motion_raw.shape[0] == 1:
                    motion_raw = motion_raw.squeeze(0)
                elif motion_raw.ndim == 2:
                    pass  # đã đúng shape
                else:
                    print(f"[WARN] skip {cid}: unexpected shape {motion_raw.shape}")
                    continue
                # if T < self.min_len or T >= 200:
                #     continue
                T, D = motion_raw.shape
                # ----- optional representation conversion -----
                betas = None         # BEAT stores body shape; keep None if you do not use it
                if self.use_rep == 'rep15d':
                    rep = get_motion_rep_numpy(motion_raw, betas=betas, device='cpu')
                    motion_proc = rep['rep15d']               # (T, 55*15)
                elif self.use_rep == 'position':
                    rep = get_motion_rep_numpy(motion_raw, betas=betas, device='cpu')
                    motion_proc = rep['position'].reshape(T, -1)
                else:  # 'axis_angle'
                    motion_proc = motion_raw                  # (T, 264)

                # ----- load caption(s) -----
                with cs.open(txt_path, 'r', encoding='utf‑8') as f:
                    caption_lines = [ln.strip() for ln in f if ln.strip()]

                for ln in caption_lines:
                    # Case A: single caption without tags
                    if '#' not in ln:
                        cdict = {'caption': ln,
                                 'tokens': ln.split(' '),
                                 'f': 0.0, 'to': 0.0}
                        self._add_example(cid, motion_proc, cdict, betas)
                    # Case B: BEAT style with time stamps
                    else:
                        try:
                            cap, toks, *tags = ln.split('#')
                            toks   = toks.strip().split(' ')
                            f_tag  = float(tags[0]) if tags else 0.0
                            to_tag = float(tags[1]) if len(tags) > 1 else 0.0
                            sub_motion = self._slice_by_time(
                                motion_proc, f_tag, to_tag
                            ) if (f_tag or to_tag) else motion_proc
                            if len(sub_motion) < self.min_len or len(sub_motion) >= 200:
                                continue
                            cdict = {'caption': cap.strip(),
                                     'tokens': toks,
                                     'f': f_tag, 'to': to_tag}
                            new_id = f'{random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ")}_{cid}'
                            self._add_example(new_id, sub_motion, cdict, betas)
                        except Exception:
                            continue
            except Exception as e:
                print(f'[WARN] skip {cid}: {e}')

        # Sort by length for easier curriculum‑style batching
        self.name_list = sorted(self.data_dict, key=lambda n: self.data_dict[n]['length'])
        self.length_arr = np.array([self.data_dict[n]['length'] for n in self.name_list])

        # -------------------------------------------------
        # 2. Stats (mean / std)
        # -------------------------------------------------
        if opt.is_train and (mean is None or std is None):
            print('Computing dataset mean / std …')
            all_motion = np.concatenate([self.data_dict[n]['motion'] for n in self.name_list], 0)
            mean = all_motion.mean(0)
            std = all_motion.std(0) + 1e-8
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)
            np.save(pjoin(opt.meta_dir, 'std.npy'),  std)
        self.mean, self.std = mean, std

    # -----------------------------------------------------
    # Helper functions
    # -----------------------------------------------------
    def _slice_by_time(self, motion, t0, t1):
        """Return sub‑sequence between seconds t0 and t1 (FPS aware)."""
        f0 = int(t0 * self.fps);  f1 = int(t1 * self.fps)
        return motion[f0:f1] if (f1 > f0) else motion

    def _add_example(self, key, motion, cdict, betas):
        self.data_dict[key] = {
            'motion':  motion,
            'length':  len(motion),
            'text':    [cdict],
            'betas':   betas
        }

    # -----------------------------------------------------
    # PyTorch Dataset API
    # -----------------------------------------------------
    def __len__(self):
        return len(self.name_list) * self.times

    def real_len(self):
        return len(self.name_list)

    def __getitem__(self, idx):
        try:
            idx %= self.real_len()
            name = self.name_list[idx]
            item = self.data_dict[name]

            motion = item['motion'].copy()                  # (T, D)
            m_len  = item['length']
            # betas  = item['betas']
            text_d = random.choice(item['text'])

            if motion is None:
                raise ValueError("motion is None")
            if not isinstance(motion, np.ndarray):
                raise TypeError(f"motion is not numpy: got {type(motion)}")
            if len(motion.shape) != 2:
                raise ValueError(f"motion shape invalid: {motion.shape}")
            if m_len != motion.shape[0]:
                raise ValueError(f"length mismatch: m_len={m_len}, motion.shape={motion.shape}")

            # crop or pad
            if m_len >= self.max_len:
                start = random.randint(0, m_len - self.max_len)
                motion = motion[start:start + self.max_len]
            else:
                pad_len = self.max_len - m_len
                pad = np.zeros((pad_len, motion.shape[1]), dtype=motion.dtype)
                motion = np.concatenate([motion, pad], 0)

            if np.any(np.isnan(motion)):
                raise ValueError("motion contains NaN")

            motion = (motion - self.mean) / self.std
            caption = text_d.get('caption', '<no caption>')
            if not isinstance(caption, str):
                raise TypeError(f"caption is not string: {caption}")
            print(f"[INFO] {name} | len: {m_len} | caption: {caption}")
            print(f"  motion shape: {motion.shape} ")
            return caption, motion.astype(np.float32), m_len

        except Exception as e:
            print("="*60)
            print(f"[ERROR @ __getitem__]")
            print(f"idx: {idx}")
            print(f"name: {self.name_list[idx]}")
            print(f"error: {e}")
            print("="*60)
            return None


        # if self.eval_mode:
        #     tokens = text_d['tokens'][:self.opt.max_text_len]
        #     tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']
        #     sent_len = len(tokens)
        #     tokens += ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)

        #     pos_ohs, w_embs = [], []
        #     for tok in tokens:
        #         w_emb, pos_oh = self.w_vectorizer[tok]
        #         pos_ohs.append(pos_oh[None])
        #         w_embs.append(w_emb[None])
        #     pos_ohs  = np.concatenate(pos_ohs, 0)
        #     w_embs   = np.concatenate(w_embs, 0)

        #     return w_embs, pos_ohs, text_d['caption'], sent_len, motion.astype(np.float32), m_len, betas

        # return text_d['caption'], motion.astype(np.float32), m_len, betas

    # -----------------------------------------------------
    # Utility
    # -----------------------------------------------------
    def inv_transform(self, data: np.ndarray):
        """Back‑transform from z‑norm to original scale."""
        return data * self.std + self.mean
