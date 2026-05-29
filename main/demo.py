#!/usr/bin/env python
"""
Usage:
    vocaset:
        python main/demo.py --config config/vocaset/demo.yaml

    BIWI:
        python main/demo.py --config config/BIWI/demo.yaml
"""

import os
import sys

import cv2
import torch
import numpy as np
import librosa
import pickle
import traceback
from transformers import Wav2Vec2Processor
import tempfile
from subprocess import call
import pyrender
import trimesh

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from base.utilities import get_parser
from models import get_model
from base.baseTrainer import load_state_dict
from style_predictor.ecapa_style_adapter import EcapaStyleAdapter

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)
cfg = get_parser()
os.environ['PYOPENGL_PLATFORM'] = 'egl' #egl


# The implementation of rendering is borrowed from VOCA: https://github.com/TimoBolkart/voca/blob/master/utils/rendering.py
def render_mesh_helper(args,
                       vertices,
                       faces,
                       t_center,
                       renderer,
                       rot=np.zeros(3),
                       z_offset=0):
    # camera params
    if args.dataset == "BIWI":
        camera_params = {
            'c': np.array([400, 400]),
            'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
            'f': np.array([4754.97941935 / 8, 4754.97941935 / 8]),
        }
    elif args.dataset == "vocaset":
        camera_params = {
            'c': np.array([400, 400]),
            'k': np.array([-0.19816071, 0.92822711, 0, 0, 0]),
            'f': np.array([4754.97941935 / 2, 4754.97941935 / 2]),
        }
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    frustum = {'near': 0.01, 'far': 3.0, 'height': 800, 'width': 800}

    # ensure numpy arrays
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)
    t_center = np.asarray(t_center, dtype=np.float32)

    # rotate around center (same math as your original)
    R = cv2.Rodrigues(rot.astype(np.float32))[0]
    v = (R @ (vertices - t_center).T).T + t_center

    intensity = 2.0
    primitive_material = pyrender.material.MetallicRoughnessMaterial(
        alphaMode='BLEND',
        baseColorFactor=[0.3, 0.3, 0.3, 1.0],
        metallicFactor=0.8,
        roughnessFactor=0.8
    )

    # trimesh -> pyrender mesh
    tri_mesh = trimesh.Trimesh(vertices=v, faces=faces, process=False)
    render_mesh = pyrender.Mesh.from_trimesh(
        tri_mesh, material=primitive_material, smooth=True
    )

    # scene
    if getattr(args, "background_black", False):
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[0, 0, 0])
    else:
        scene = pyrender.Scene(ambient_light=[.2, .2, .2], bg_color=[255, 255, 255])

    camera = pyrender.IntrinsicsCamera(
        fx=float(camera_params['f'][0]),
        fy=float(camera_params['f'][1]),
        cx=float(camera_params['c'][0]),
        cy=float(camera_params['c'][1]),
        znear=frustum['near'],
        zfar=frustum['far']
    )

    scene.add(render_mesh, pose=np.eye(4))

    # camera pose (use your original)
    scene.add(camera, pose=np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1.0 - float(z_offset)],
        [0, 0, 0, 1],
    ], dtype=np.float32))

    # lights (same as your original)
    angle = np.pi / 6.0
    pos = np.array([0, 0, 1.0 - float(z_offset)], dtype=np.float32)

    light_color = np.array([1., 1., 1.], dtype=np.float32)
    light = pyrender.DirectionalLight(color=light_color, intensity=intensity)

    def add_light(p):
        lp = np.eye(4, dtype=np.float32)
        lp[:3, 3] = p
        scene.add(light, pose=lp)

    add_light(pos)
    add_light(cv2.Rodrigues(np.array([ angle, 0, 0], dtype=np.float32))[0].dot(pos))
    add_light(cv2.Rodrigues(np.array([-angle, 0, 0], dtype=np.float32))[0].dot(pos))
    add_light(cv2.Rodrigues(np.array([0, -angle, 0], dtype=np.float32))[0].dot(pos))
    add_light(cv2.Rodrigues(np.array([0,  angle, 0], dtype=np.float32))[0].dot(pos))

    # render (no new OffscreenRenderer here!)
    flags = pyrender.RenderFlags.SKIP_CULL_FACES
    try:
        color, _ = renderer.render(scene, flags=flags)
    except Exception as e:
        print("pyrender render exception:", repr(e))
        traceback.print_exc()
        color = np.zeros((frustum['height'], frustum['width'], 3), dtype='uint8')

    return color[..., ::-1]


def main():
    global cfg
    model = get_model(cfg)
    # if torch.cuda.is_available():
    model = model.cuda()

    if os.path.isfile(cfg.model_path):
        print("=> loading checkpoint '{}'".format(cfg.model_path))
        checkpoint = torch.load(cfg.model_path, map_location=lambda storage, loc: storage.cpu())
        load_state_dict(model, checkpoint['state_dict'], strict=False)
        print("=> loaded checkpoint '{}'".format(cfg.model_path))
    else:
        raise RuntimeError("=> no checkpoint flound at '{}'".format(cfg.model_path))

    model.eval()
    save_folder = cfg.demo_npy_save_folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    condition = cfg.condition
    subject = cfg.subject
    test(model, cfg.demo_wav_path, save_folder, condition, subject)


def test(model, wav_file, save_folder, condition, subject):
    # generate the facial animation (.npy file) for the audio 
    print('Generating facial animation for {}...'.format(wav_file))
    
    template_file = os.path.join(cfg.data_root, cfg.template_file)
    with open(template_file, 'rb') as fin:
        templates = pickle.load(fin,encoding='latin1')
    

    # train_subjects_list = [i for i in cfg.train_subjects.split(" ")]
    # one_hot_labels = np.eye(len(train_subjects_list))
    # iter = train_subjects_list.index(condition)
    # one_hot = one_hot_labels[iter]
    # one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
    # one_hot = torch.FloatTensor(one_hot).to(device='cuda')

    # ========================= 新增：风格预测器 =========================
    gallery_path = "StylePredictor/checkpoint/ecapa_gallery_proj_S8.npy" 
    ckpt_path = "StylePredictor/checkpoint/ecapa_proj_decontent_S8.pth"

    adapter = EcapaStyleAdapter(
        gallery_path=gallery_path,
        ckpt_path=ckpt_path,
        out_dim=len(cfg.train_subjects.split(" ")),
        temperature=0.2, # 越小越趋近于硬分类，越大混合得越平均
        device="cuda",
    )

    mixed_one_hot, info = adapter.predict_mixed_one_hot_from_wav_path(
        wav_file,
        num_views=10, 
        crop_len=32000,
        topk=5
    )

    print(f"\n[Style Predictor] Result for: {os.path.basename(wav_file)}")
    print(f" >> Most similar style: {info['top1_id']} ({info['top1_prob']:.2%})")
    print(f" >> Top 5 Mix: {info['topk']}")

    one_hot = mixed_one_hot.to(device='cuda') 
    # ------------------------------------------------------------

    temp = templates[subject]
    template = temp.reshape((-1))
    template = np.reshape(template,(-1,template.shape[0]))
    template = torch.FloatTensor(template).to(device='cuda')


    test_name = os.path.basename(wav_file).split(".")[0]
    if not os.path.exists(os.path.join(save_folder,test_name)):
        os.makedirs(os.path.join(save_folder,test_name))
    predicted_vertices_path = os.path.join(save_folder, test_name, 'condition_'+condition+'_subject_'+subject+'.npy')
    speech_array, _ = librosa.load(wav_file, sr=16000)
    _local = os.path.isdir(cfg.wav2vec2model_path)
    processor = Wav2Vec2Processor.from_pretrained(cfg.wav2vec2model_path, local_files_only=_local)
    audio_feature = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
    audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
    audio_feature = torch.FloatTensor(audio_feature).to(device='cuda')



    with torch.no_grad():
        prediction = model.predict(audio_feature, template, one_hot)
        prediction = prediction.squeeze() # (seq_len, V*3)
        np.save(predicted_vertices_path, prediction.detach().cpu().numpy())
        print(f'Save facial animation in {predicted_vertices_path}')


    ######################################################################################
    ##### render the npy file

    if cfg.dataset == "BIWI":
        template_file = os.path.join(cfg.data_root, "BIWI.ply")
    elif cfg.dataset == "vocaset":
        template_file = os.path.join(cfg.data_root, "FLAME_sample.ply")
         
    print("rendering: ", test_name)
                 
    # template = Mesh(filename=template_file)
    template = trimesh.load(template_file, process=False)  # 读ply
    faces = np.asarray(template.faces)
    renderer = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=800)

    predicted_vertices = np.load(predicted_vertices_path)
    predicted_vertices = np.reshape(predicted_vertices,(-1,cfg.vertice_dim//3,3))

    output_path = cfg.demo_output_path
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    num_frames = predicted_vertices.shape[0]
    tmp_video_file = tempfile.NamedTemporaryFile('w', suffix='.mp4', dir=output_path)
    
    writer = cv2.VideoWriter(tmp_video_file.name, cv2.VideoWriter_fourcc(*'mp4v'), cfg.fps, (800, 800), True)
    center = np.mean(predicted_vertices[0], axis=0)

    for i_frame in range(num_frames):
        # render_mesh = Mesh(predicted_vertices[i_frame], template.f)
        # pred_img = render_mesh_helper(cfg,render_mesh, center)
        vertices = predicted_vertices[i_frame]
        pred_img = render_mesh_helper(cfg, vertices, faces, center, renderer=renderer)
        pred_img = pred_img.astype(np.uint8)
        writer.write(pred_img)

    writer.release()
    file_name = test_name+"_"+cfg.subject+"_condition_"+cfg.condition

    video_fname = os.path.join(output_path, file_name+'.mp4')
    cmd = ('ffmpeg' + ' -i {0} -pix_fmt yuv420p -qscale 0 {1}'.format(
       tmp_video_file.name, video_fname)).split()
    call(cmd)

    cmd = ('ffmpeg' + ' -i {0} -i {1} -vcodec h264 -ac 2 -channel_layout stereo -qscale 0 {2}'.format(
       wav_file, video_fname, video_fname.replace('.mp4', '_audio.mp4'))).split()
    call(cmd)

    if os.path.exists(video_fname):
        os.remove(video_fname)

if __name__ == '__main__':
    main()
