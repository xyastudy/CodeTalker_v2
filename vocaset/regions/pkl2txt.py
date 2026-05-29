import pickle
import json
import numpy as np
import os

# 1. 加载 pkl 文件
pkl_path = '/home/an/code/CodeTalker_v2/vocaset/regions/FLAME_masks.pkl'
with open(pkl_path, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

txt_save_path = '/home/an/code/CodeTalker_v2/vocaset/regions/readable_data.txt'

def format_list_20(lst, indent_level):
    """处理列表：每 20 个元素换行，并保持缩进"""
    spacing = " " * indent_level
    rows = []
    for i in range(0, len(lst), 20):
        # 取 20 个元素并转为字符串
        chunk = ", ".join(map(str, lst[i:i+20]))
        rows.append(f"{spacing}    {chunk}")
    return "[\n" + ",\n".join(rows) + "\n" + spacing + "]"

# 2. 写入 TXT (手动控制排序与换行)
with open(txt_save_path, 'w', encoding='utf-8') as f:
    f.write("{\n")
    
    # 对字典的 key 进行字母顺序排序
    sorted_keys = sorted(data.keys())
    
    for i, key in enumerate(sorted_keys):
        value = data[key]
        # 写入 Key
        f.write(f'    "{key}": ')
        
        # 处理 Value (列表)
        if isinstance(value, (list, np.ndarray, tuple)):
            lst = value.tolist() if hasattr(value, 'tolist') else list(value)
            # 调用每 20 个换行的函数 (缩进为 4)
            f.write(format_list_20(lst, 4))
        else:
            # 处理非列表数据 (如 count 等)
            f.write(json.dumps(value))
        
        # 处理逗号 (最后一个元素不加逗号)
        if i < len(sorted_keys) - 1:
            f.write(",\n")
        else:
            f.write("\n")
            
    f.write("}")

print(f"✅ 排序完成！关键字已按 A-Z 排列，且列表已设为每 20 个元素换行。")