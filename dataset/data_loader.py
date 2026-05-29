import os
import torch
import numpy as np
import pickle
from tqdm import tqdm
from collections import defaultdict
from torch.utils import data 
from torch.nn.utils.rnn import pad_sequence

class Dataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader."""
    def __init__(self, data,subjects_dict,data_type="train",read_audio=False):
        self.data = data
        self.len = len(self.data)
        self.subjects_dict = subjects_dict
        self.data_type = data_type
        self.one_hot_labels = np.eye(len(subjects_dict["train"]))
        self.read_audio = read_audio

    def __getitem__(self, index):
        """Returns one data pair (source and target)."""
        # seq_len, fea_dim
        file_name = self.data[index]["name"]
        audio = self.data[index]["audio"]
        vertice = self.data[index]["vertice"]
        template = self.data[index]["template"]
        if self.data_type == "train":
            subject = "_".join(file_name.split("_")[:-1])
            one_hot = self.one_hot_labels[self.subjects_dict["train"].index(subject)]
        else:
            one_hot = self.one_hot_labels
        if self.read_audio:
            return torch.FloatTensor(audio),torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name
        else:
            return torch.FloatTensor(vertice), torch.FloatTensor(template), torch.FloatTensor(one_hot), file_name

    def __len__(self):
        return self.len

def read_data(args):
    print("Loading data...")
    data = defaultdict(dict)
    train_data = []
    valid_data = []
    test_data = []

    audio_path = os.path.join(args.data_root, args.wav_path)
    vertices_path = os.path.join(args.data_root, args.vertices_path)
    if args.read_audio: # read_audio==False when training vq to save time
        from transformers import Wav2Vec2Processor
        import librosa
        processor = Wav2Vec2Processor.from_pretrained(args.wav2vec2model_path)

    template_file = os.path.join(args.data_root, args.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')
    
    for r, ds, fs in os.walk(audio_path):
        for f in tqdm(fs):
            if f.endswith("wav"):
                if args.read_audio:
                    wav_path = os.path.join(r,f)
                    speech_array, sampling_rate = librosa.load(wav_path, sr=16000)
                    input_values = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
                key = f.replace("wav", "npy")
                data[key]["audio"] = input_values if args.read_audio else None
                subject_id = "_".join(key.split("_")[:-1])
                temp = templates[subject_id]
                data[key]["name"] = f
                data[key]["template"] = temp.reshape((-1)) 
                vertice_path = os.path.join(vertices_path,f.replace("wav", "npy"))
                if not os.path.exists(vertice_path):
                    del data[key]
                else:
                    if args.dataset == "vocaset":
                        data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)[::2,:]#due to the memory limit
                    elif args.dataset == "BIWI":
                        data[key]["vertice"] = np.load(vertice_path,allow_pickle=True)

    subjects_dict = {}
    subjects_dict["train"] = [i for i in args.train_subjects.split(" ")]
    subjects_dict["val"] = [i for i in args.val_subjects.split(" ")]
    subjects_dict["test"] = [i for i in args.test_subjects.split(" ")]


    #train vq and pred
    splits = {'vocaset':{'train':range(1,41),'val':range(21,41),'test':range(21,41)},
    'BIWI':{'train':range(1,33),'val':range(33,37),'test':range(37,41)}}


    for k, v in data.items():
        subject_id = "_".join(k.split("_")[:-1])
        sentence_id = int(k.split(".")[0][-2:])
        if subject_id in subjects_dict["train"] and sentence_id in splits[args.dataset]['train']:
            train_data.append(v)
        if subject_id in subjects_dict["val"] and sentence_id in splits[args.dataset]['val']:
            valid_data.append(v)
        if subject_id in subjects_dict["test"] and sentence_id in splits[args.dataset]['test']:
            test_data.append(v)

    print('Loaded data: Train-{}, Val-{}, Test-{}'.format(len(train_data), len(valid_data), len(test_data)))
    return train_data, valid_data, test_data, subjects_dict

def collate_fn_read_audio(batch):
    """
    batch: List[(audio[T], vertice[T_v,D], template[D_t], one_hot[N], file_name)]
    """
    audios, vertices, templates, one_hots, file_names = zip(*batch)

    # 确保连续内存（避免奇怪的 storage/stride 问题）
    audios   = [a.float().contiguous() for a in audios]       # [T]
    vertices = [v.float().contiguous() for v in vertices]     # [T_v, D]

    audio_lens   = torch.tensor([a.shape[0] for a in audios], dtype=torch.long)
    vertice_lens = torch.tensor([v.shape[0] for v in vertices], dtype=torch.long)

    # padding
    audios   = pad_sequence(audios, batch_first=True, padding_value=0.0)      # [B, T_a_max]
    vertices = pad_sequence(vertices, batch_first=True, padding_value=0.0)    # [B, T_v_max, D]

    templates = torch.stack([t.float() for t in templates], dim=0)            # [B, D_t]
    one_hots  = torch.stack([o.float() for o in one_hots], dim=0)             # [B, N]

    return audios, audio_lens, vertices, vertice_lens, templates, one_hots, file_names

def collate_fn_no_audio(batch):
    """
    batch: List[(vertice[T_v,D], template[D_t], one_hot[N], file_name)]
    """
    vertices, templates, one_hots, file_names = zip(*batch)

    vertices = [v.float().contiguous() for v in vertices]
    vertice_lens = torch.tensor([v.shape[0] for v in vertices], dtype=torch.long)

    vertices  = pad_sequence(vertices, batch_first=True, padding_value=0.0)   # [B, T_v_max, D]
    templates = torch.stack([t.float() for t in templates], dim=0)            # [B, D_t]
    one_hots  = torch.stack([o.float() for o in one_hots], dim=0)             # [B, N]

    return vertices, vertice_lens, templates, one_hots, file_names

def get_dataloaders(args):
    """
    if read_audio = true:
        return (audio, audio_lens, vertice, vertice_lens, template, one_hot_all, file_name)
    else:
        return (vertice, vertice_lens, template, one_hot_all, file_name)

    DDP 模式下自动使用 DistributedSampler 对训练集分片，返回的 dict 中额外包含
    'train_sampler' 键（单卡时为 None），训练脚本需在每个 epoch 开始前调用
    sampler.set_epoch(epoch) 以保证各卡 shuffle 顺序不同。
    """
    dataset = {}
    train_data, valid_data, test_data, subjects_dict = read_data(args)

    train_set = Dataset(train_data, subjects_dict, "train", args.read_audio)
    valid_set = Dataset(valid_data, subjects_dict, "val", args.read_audio)
    test_set  = Dataset(test_data,  subjects_dict, "test", args.read_audio)

    collate = collate_fn_read_audio if args.read_audio else collate_fn_no_audio

    # DDP：用 DistributedSampler 把训练集均匀分配给每张卡，避免重复计算
    distributed = getattr(args, 'distributed', False)
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        train_sampler = DistributedSampler(train_set, shuffle=True)
        train_shuffle = False   # shuffle 由 sampler 负责
    else:
        train_sampler = None
        train_shuffle = True

    dataset["train"] = data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=True,
    )
    dataset["train_sampler"] = train_sampler  # None when single-GPU

    dataset["valid"] = data.DataLoader(
        dataset=valid_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate,
        pin_memory=True,
    )

    dataset["test"] = data.DataLoader(
        dataset=test_set,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate,
    )

    return dataset

if __name__ == "__main__":
    get_dataloaders()