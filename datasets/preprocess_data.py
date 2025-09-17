from pymo.parsers import BVHParser
from pymo.viz_tools import *
from pymo.preprocessing import *
from sklearn.pipeline import Pipeline
import tqdm
import os  
import numpy as np  
from textgrid import TextGrid 

  
def time_to_frame(t, fps=60):  
    return int(round(t * fps))

def extract_sentences_with_text(textgrid_path, motion_data, output_dir, fps=30, pause_threshold=0.5,
    split_parts=1, use_first_part_only=True
):
    basename = os.path.splitext(os.path.basename(textgrid_path))[0]
    print(motion_data.shape)
    if motion_data.ndim == 3:  # Shape: (1, frames, features)  
        split_data = np.split(motion_data, split_parts, axis=1)  
        
        if use_first_part_only:  
            motion = split_data[0].squeeze(0)    
        else:  
            reshaped_data = [data.squeeze(0) for data in split_data]  
            motion = reshaped_data[0]  # Sử dụng phần đầu tiên  
    else:  
        # Nếu đã là 2D, sử dụng trực tiếp  
        motion = motion_data[0] if motion_data.ndim == 2 else motion_data  
      
    max_frames = motion.shape[0]  
      
    tg = TextGrid.fromFile(textgrid_path)  
    tier = tg[0]  
  
    # with open(text_path, 'r') as f:  
    #     full_text = f.read().strip()  
  
    os.makedirs(output_dir, exist_ok=True)  
  
    sentence_start = None  
    sentence_end = None  
    sentence_text = []  
    sentence_idx = 0  
      
    def save_sentence():  
        nonlocal sentence_idx, sentence_start, sentence_end, sentence_text  
          
        if not sentence_text or sentence_start is None or sentence_end is None:  
            return  
              
        start_frame = time_to_frame(sentence_start, fps)  
        end_frame = time_to_frame(sentence_end, fps)  
          
        # Đảm bảo không vượt quá bounds của motion data  
        start_frame = max(0, start_frame)  
        end_frame = min(max_frames, end_frame)  
          
        if end_frame <= start_frame:  
            print(f"⚠️ Skipped sentence {sentence_idx}: invalid frame range [{start_frame}, {end_frame}]")  
            return  
              
        motion_segment = motion[start_frame:end_frame, :]  
          
        fname_base = f"{basename}_sentence_{sentence_idx:03d}"  
        np.save(os.path.join(output_dir, fname_base + ".npy"), motion_segment)  
          
        with open(os.path.join(output_dir, fname_base + ".txt"), 'w') as ftxt:  
            ftxt.write(" ".join(sentence_text))  
          
        print(f"✅ Saved: {fname_base}.npy & .txt (frames: {start_frame}-{end_frame}, shape: {motion_segment.shape})")  
        sentence_idx += 1  
  
    for interval in tier.intervals:  
        word = interval.mark.strip()  
        xmin = float(interval.minTime)  
        xmax = float(interval.maxTime)  
  
        if word != "":  
            if sentence_start is None:  
                sentence_start = xmin  
            sentence_end = xmax  
            sentence_text.append(word)  
        else:  
            pause_duration = xmax - xmin  
            if pause_duration >= pause_threshold and sentence_text:  
                save_sentence()  
                  
                # Reset cho câu tiếp theo  
                sentence_start = None  
                sentence_end = None  
                sentence_text = []  
  
    # Lưu câu cuối cùng  
    if sentence_text:  
        save_sentence()  
  
    print(f"🎉 Extracted {sentence_idx} sentences from {basename}")

def preprocess_motion_data(base_dir, npy_out_dir, txt_out_dir):
    parser = BVHParser()

    data_pipe = Pipeline([
        ('param', MocapParameterizer('position')),
        ('rcpn', RootCentricPositionNormalizer()),
        ('delta', RootTransformer('abdolute_translation_deltas')),
        ('const', ConstantsRemover()),
        ('np', Numpyfier()),
        ('down', DownSampler(2)),
        ('stdscale', ListStandardScaler())
    ])

    for fname in os.listdir(base_dir):
        if fname.endswith(".bvh"):
            basename = fname.replace(".bvh", "")
            bvh_path = os.path.join(base_dir, fname)
            textgrid_path = os.path.join(base_dir, basename + ".TextGrid")
            text_path = os.path.join(base_dir, basename + ".txt")
            if not os.path.exists(textgrid_path) or not os.path.exists(text_path):
                continue  # Bỏ qua nếu thiếu file

            # Parse BVH và chạy pipeline
            if not os.path.exists(textgrid_path) or not os.path.exists(text_path):
                continue  # Bỏ qua nếu thiếu file

            try:
                parsed_data = parser.parse(bvh_path)
            except Exception as e:
                print(f"❌ Lỗi khi parse {bvh_path}: {e}")
                continue
            piped_data = data_pipe.fit_transform([parsed_data])

            # Gọi hàm tách segment
            extract_sentences_with_text(
                textgrid_path=textgrid_path,
                text_path=text_path,
                motion_data=piped_data,
                output_dir=npy_out_dir,
                split_parts=1,
                use_first_part_only=True
            )

            # Di chuyển file .txt sang txt_out_dir
            for f in os.listdir(npy_out_dir):
                if f.endswith(".txt"):
                    os.rename(os.path.join(npy_out_dir, f), os.path.join(txt_out_dir, f))

def main():
    base_dir = "data/bvh"
    npy_out_dir = "data/npy"
    txt_out_dir = "data/txt"

    os.makedirs(npy_out_dir, exist_ok=True)
    os.makedirs(txt_out_dir, exist_ok=True)

    preprocess_motion_data(base_dir, npy_out_dir, txt_out_dir)