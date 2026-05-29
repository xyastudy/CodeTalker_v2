import os
import sys
from datetime import datetime

project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from main.train_vq import main as train


def main():
    now = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_name = f"triple_codebook_{now}"
    cfg_path = os.path.join(project_root, "config/vocaset/stage1.yaml")  
    save_path = os.path.join(project_root, f"RUN/vocaset/s1/{exp_name}")

    opts = ["save_path", save_path]

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["OMP_NUM_THREADS"] = "10"
    os.environ["PYTHONPATH"] = "./"

    print(f"####### Starting Stage 1 Training...")
    print(f"####### Config: {cfg_path}")
    print(f"####### Saving to: {save_path}")

    try:
        train(cfg_path, opts) 
    except Exception as e:
        print(f"Training Interrupted: {e}")
        raise e

if __name__ == "__main__":
    main()