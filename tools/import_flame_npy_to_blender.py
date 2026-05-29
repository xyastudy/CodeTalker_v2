import bpy
import numpy as np

# npy 文件路径
npy_path = "/home/an/code/CodeTalker_v2/RUN/vocaset/CodeTalkerV2_s2/result/npy/FaceTalk_170731_00024_TA_sentence21_condition_FaceTalk_170725_00137_TA.npy"
# 预期形状为 (frames, 顶点数*3) 或 (frames, 顶点数, 3)
data = np.load(npy_path)
num_frames = data.shape[0]

if len(data.shape) == 2:
    data = data.reshape(num_frames, -1, 3)

obj = bpy.context.active_object
mesh = obj.data

for f in range(num_frames):
    bpy.context.scene.frame_set(f)
    frame_coords = data[f]

    for i, v in enumerate(mesh.vertices):
        v.co = frame_coords[i]
        v.keyframe_insert(data_path="co", frame=f)

    if (f + 1) % 10 == 0:
        print(f"已加载 {f + 1}/{num_frames} 帧")

print(f"成功加载 {num_frames} 帧动画!!!")
