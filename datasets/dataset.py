import torch  
from torch.utils import data  
import numpy as np  
import os  
from os.path import join as pjoin  
import random  
import codecs as cs  
from tqdm import tqdm  
from utils.motion_io import beat_format_load  
from utils.motion_rep_transfer import get_motion_rep_numpy  
  
class Beat2MotionDataset(data.Dataset):  
    """Dataset for BEAT Motion generation task with diffusion models."""  
      
    def __init__(self, opt, mean, std, split_file, times=1, w_vectorizer=None, eval_mode=False):  
        self.opt = opt  
        self.max_length = 20  
        self.times = times  
        self.w_vectorizer = w_vectorizer  
        self.eval_mode = eval_mode  
        min_motion_len = 40 if self.opt.dataset_name == 'beat' else 24  
          
        # BEAT dataset sử dụng 55 joints  
        joints_num = 55  
          
        data_dict = {}  
        id_list = []  
        with cs.open(split_file, 'r') as f:  
            for line in f.readlines():  
                id_list.append(line.strip())  
          
        new_name_list = []  
        length_list = []  
        for name in tqdm(id_list):  
            try:  
                # Load BEAT format thay vì .npy  
                motion_dict = beat_format_load(pjoin(opt.motion_dir, name + '.npz'))  
                motion = motion_dict['poses']  # Shape: (t, 165) - axis-angle cho 55 joints  
                betas = motion_dict['betas']  
                  
                if len(motion) < min_motion_len or len(motion) >= 200:  
                    continue  
                  
                # Convert sang representation phù hợp cho diffusion  
                if opt.motion_rep == 'rep15d':  
                    motion_rep = get_motion_rep_numpy(motion, device="cpu", betas=betas)  
                    motion = motion_rep['rep15d']  # Shape: (t, 55*15)  
                elif opt.motion_rep == 'position':  
                    motion_rep = get_motion_rep_numpy(motion, device="cpu", betas=betas)  
                    motion = motion_rep['position'].reshape(len(motion), -1)  # Shape: (t, 55*3)  
                elif opt.motion_rep == 'axis_angle':  
                    motion = motion  # Giữ nguyên axis-angle format (t, 165)  
                  
                text_data = []  
                flag = False  
                with cs.open(pjoin(opt.text_dir, name + '.txt')) as f:  
                    for line in f.readlines():  
                        text_dict = {}  
                        line_split = line.strip().split('#')  
                        caption = line_split[0]  
                        tokens = line_split[1].split(' ')  
                        f_tag = float(line_split[2]) if len(line_split) > 2 else 0.0  
                        to_tag = float(line_split[3]) if len(line_split) > 3 else 0.0  
                        f_tag = 0.0 if np.isnan(f_tag) else f_tag  
                        to_tag = 0.0 if np.isnan(to_tag) else to_tag  
  
                        text_dict['caption'] = caption  
                        text_dict['tokens'] = tokens  
                        if f_tag == 0.0 and to_tag == 0.0:  
                            flag = True  
                            text_data.append(text_dict)  
                        else:  
                            # Segment motion theo timestamps  
                            start_frame = int(f_tag * 30)  # BEAT dataset 30fps  
                            end_frame = int(to_tag * 30)  
                            n_motion = motion[start_frame:end_frame]  
                            if len(n_motion) < min_motion_len or len(n_motion) >= 200:  
                                continue  
                            new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name  
                            while new_name in data_dict:  
                                new_name = random.choice('ABCDEFGHIJKLMNOPQRSTUVW') + '_' + name  
                            data_dict[new_name] = {  
                                'motion': n_motion,  
                                'length': len(n_motion),  
                                'text': [text_dict],  
                                'betas': betas  
                            }  
                            new_name_list.append(new_name)  
                            length_list.append(len(n_motion))  
  
                if flag:  
                    data_dict[name] = {  
                        'motion': motion,  
                        'length': len(motion),  
                        'text': text_data,  
                        'betas': betas  
                    }  
                    new_name_list.append(name)  
                    length_list.append(len(motion))  
            except Exception as e:  
                print(f"Error loading {name}: {e}")  
                pass  
  
        name_list, length_list = zip(*sorted(zip(new_name_list, length_list), key=lambda x: x[1]))  
  
        # Simple normalization cho BEAT dataset  
        if opt.is_train:  
            np.save(pjoin(opt.meta_dir, 'mean.npy'), mean)  
            np.save(pjoin(opt.meta_dir, 'std.npy'), std)  
  
        self.mean = mean  
        self.std = std  
        self.length_arr = np.array(length_list)  
        self.data_dict = data_dict  
        self.name_list = name_list  
  
    def inv_transform(self, data):  
        return data * self.std + self.mean  
  
    def real_len(self):  
        return len(self.data_dict)  
  
    def __len__(self):  
        return self.real_len() * self.times  
  
    def __getitem__(self, item):  
        idx = item % self.real_len()  
        data = self.data_dict[self.name_list[idx]]  
        motion, m_length, text_list = data['motion'], data['length'], data['text']  
        betas = data['betas']  
          
        # Randomly select a caption  
        text_data = random.choice(text_list)  
        caption = text_data['caption']  
  
        max_motion_length = self.opt.max_motion_length  
        if m_length >= max_motion_length:  
            idx = random.randint(0, len(motion) - max_motion_length)  
            motion = motion[idx: idx + max_motion_length]  
        else:  
            padding_len = max_motion_length - m_length  
            D = motion.shape[1]  
            padding_zeros = np.zeros((padding_len, D))  
            motion = np.concatenate((motion, padding_zeros), axis=0)  
  
        assert len(motion) == max_motion_length  
          
        # Z Normalization  
        motion = (motion - self.mean) / self.std  
  
        if self.eval_mode:  
            tokens = text_data['tokens']  
            if len(tokens) < self.opt.max_text_len:  
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']  
                sent_len = len(tokens)  
                tokens = tokens + ['unk/OTHER'] * (self.opt.max_text_len + 2 - sent_len)  
            else:  
                tokens = tokens[:self.opt.max_text_len]  
                tokens = ['sos/OTHER'] + tokens + ['eos/OTHER']  
                sent_len = len(tokens)  
              
            pos_one_hots = []  
            word_embeddings = []  
            for token in tokens:  
                word_emb, pos_oh = self.w_vectorizer[token]  
                pos_one_hots.append(pos_oh[None, :])  
                word_embeddings.append(word_emb[None, :])  
            pos_one_hots = np.concatenate(pos_one_hots, axis=0)  
            word_embeddings = np.concatenate(word_embeddings, axis=0)  
            return word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, betas  
          
        return caption, motion, m_length, betas