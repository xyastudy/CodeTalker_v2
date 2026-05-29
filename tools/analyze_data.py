import os

from collections import Counter
import soundfile as sf
import numpy as np

wav_folder_path = "/home/an/code/CodeTalker/vocaset/wav" 

def analyze_subjects(path):
    if not os.path.exists(path):
        print(f"错误：路径 {path} 不存在。")
        return

    files = [f for f in os.listdir(path) if f.endswith('.wav')]

    subjects = []
    durations = []      # 秒
    lengths = []        # 采样点数
    sr_set = set()      # 采样率（顺便检查是否一致）

    for f in files:
        name = os.path.splitext(f)[0]
        parts = name.split('_')

        if len(parts) >= 3:
            subject_id = f"{parts[1]}_{parts[2]}"
            subjects.append(subject_id)

        wav_path = os.path.join(path, f)
        try:
            info = sf.info(wav_path)
            dur = info.frames / info.samplerate
            durations.append(dur)
            lengths.append(info.frames)
            sr_set.add(info.samplerate)
        except Exception as e:
            print(f"读取失败: {f}, error={e}")

    counts = Counter(subjects)

    print("=" * 40)
    print(f"分析路径: {path}")
    print(f"总文件数: {len(files)}")
    print(f"唯一采集者数量: {len(counts)}")
    print(f"采样率集合: {sorted(list(sr_set))}")
    print("=" * 40)

    print("【每个采集者的句子数】")
    for sub, c in counts.most_common():
        print(f"ID: {sub} | 句子数量: {c}")

    if durations:
        print("=" * 40)
        print("【音频时长统计（秒）】")
        print(f"最短: {min(durations):.3f}")
        print(f"最长: {max(durations):.3f}")
        print(f"平均: {np.mean(durations):.3f}")
        print(f"中位数: {np.median(durations):.3f}")

        print("=" * 40)
        print("【最常见的 10 种时长（四舍五入到 0.1s）】")
        rounded = [round(d, 1) for d in durations]
        for d, c in Counter(rounded).most_common(10):
            print(f"{d:.1f}s : {c}")

        print("=" * 40)
        print("【最常见的 10 种采样点长度】")
        for l, c in Counter(lengths).most_common(10):
            print(f"{l} samples : {c}")


if __name__ == "__main__":
    analyze_subjects(wav_folder_path)

