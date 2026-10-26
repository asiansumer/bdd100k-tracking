"""第 6 关：用三种运动模型各跑一遍跟踪，结果写成 MOT 格式，给 TrackEval 评估。

运行：python -m scripts.run_track
结果：results/<模型名>/data/<视频名>.txt，每行 帧号,编号,左上x,左上y,宽,高,1,-1,-1,-1
"""
import os

from bdd_io import load_dets
from motion_models import NoMotion, AlphaBeta, Kalman
from scripts.prepare_bdd import VIDEOS, DATA
from tracker import Tracker

MODELS = {
    "no_motion": NoMotion,
    "alpha_beta": AlphaBeta,
    "kalman": Kalman,
}


def run_video(video, motion, out_dir):
    dets = load_dets(f"dets/{video}.txt")
    n_frames = len(os.listdir(f"{DATA}/seqs/{video}/img1"))
    tracker = Tracker(motion)
    with open(f"{out_dir}/{video}.txt", "w") as f:
        for frame in range(1, n_frames + 1):
            for tid, (x1, y1, x2, y2) in tracker.step(dets.get(frame, [])):
                f.write(f"{frame},{tid},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},1,-1,-1,-1\n")
    return tracker.next_id - 1


if __name__ == "__main__":
    for name, Model in MODELS.items():
        out_dir = f"results/{name}/data"
        os.makedirs(out_dir, exist_ok=True)
        total = 0
        for v in VIDEOS:
            total += run_video(v, Model(), out_dir)
        print(f"{name}: 10 段视频一共发了 {total} 个编号")
