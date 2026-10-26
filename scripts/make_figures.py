"""第 7 关：画 README 用的图，都存到 figures/。

运行：python -m scripts.make_figures            （全画）
     python -m scripts.make_figures compare     （只画某一张：compare / frame_iou / align / gif）
"""
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from bdd_io import load_dets, load_ped_labels, read_bdd_parquet
from box_utils import iou
from motion_models import Kalman
from scripts.prepare_bdd import DATA, MODEL_PATH, VIDEOS, find_offset, read_frames, video_index
from tracker import Tracker

OUT = "figures"
NAMES = ["no_motion", "alpha_beta", "kalman"]
LABELS = {"no_motion": "No motion", "alpha_beta": "Alpha-beta", "kalman": "Kalman"}
COLORS = {"no_motion": "#2a78d6", "alpha_beta": "#eb6834", "kalman": "#1baf7a"}
# 动图用第 1 段（第五大道路口），从第 110 帧开始，行人最多
GIF_VIDEO = "00abd8a7-ecd6fc56"
GIF_START = 110


def read_summary(name):
    """读 TrackEval 写的 results/<名字>/pedestrian_summary.txt，返回 {指标: 数}。"""
    with open(f"results/{name}/pedestrian_summary.txt") as f:
        keys = f.readline().split()
        values = [float(x) for x in f.readline().split()]
    return dict(zip(keys, values))


def fig_compare():
    """三种运动模型的 HOTA / MOTA / IDF1 柱状图。"""
    metrics = ["HOTA", "MOTA", "IDF1"]
    fig, ax = plt.subplots(figsize=(7, 4))
    width = 0.26
    for k, name in enumerate(NAMES):
        s = read_summary(name)
        xs = np.arange(len(metrics)) + (k - 1) * width
        ys = [s[m] for m in metrics]
        ax.bar(xs, ys, width - 0.03, color=COLORS[name], label=LABELS[name])
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.6, f"{y:.1f}", ha="center", fontsize=9, color="#333333")
    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics)
    ax.set_ylabel("score (%)")
    ax.set_title("Same detector, same tracker, three motion models (10 BDD100K videos)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5")
    ax.set_axisbelow(True)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/compare.png", dpi=150)
    print("figures/compare.png")


def fig_frame_iou():
    """同一个人前后两个标注帧的框 IoU 分布：5 帧/秒 + 车在动，很多几乎不重叠。"""
    ious = []
    for v in VIDEOS:
        tracks = {}
        for line in open(f"{DATA}/seqs/{v}/gt/gt.txt"):
            p = line.split(",")
            frame, tid = int(p[0]), int(p[1])
            x, y, w, h = map(float, p[2:6])
            tracks.setdefault(tid, {})[frame] = (x, y, x + w, y + h)
        for t in tracks.values():
            for frame in t:
                if frame + 1 in t:
                    ious.append(iou(t[frame], t[frame + 1]))
    ious = np.array(ious)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(ious, bins=20, range=(0, 1), color="#2a78d6", edgecolor="white", linewidth=2)
    low = np.mean(ious < 0.3) * 100
    ax.axvline(0.3, color="#555555", linestyle="--", linewidth=1)
    ax.text(0.31, ax.get_ylim()[1] * 0.9, f"{low:.0f}% below IoU 0.3", color="#333333")
    ax.set_xlabel("IoU of the same pedestrian between two consecutive labeled frames (5 fps)")
    ax.set_ylabel("count")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{OUT}/frame_iou.png", dpi=150)
    print("figures/frame_iou.png", f"中位数 {np.median(ious):.2f}，低于 0.3 的占 {low:.0f}%")


def draw_boxes(img, boxes, color, ids=None, font=0.6):
    for k, (x1, y1, x2, y2) in enumerate(boxes):
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        if ids is not None:
            cv2.putText(img, str(ids[k]), (int(x1), int(y1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, font, color, 2)
    return img


def fig_align(video=VIDEOS[0]):
    """同一个标注帧，按时间直接换算（偏移 0）和找到的偏移各抽一张，画上标注框对比。"""
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    df = read_bdd_parquet(f"{DATA}/mot_labels.parquet")
    labels = load_ped_labels(df, video)
    path = f"{DATA}/videos/{video}.mov"
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    offset, score = find_offset(path, labels, fps, model)
    # 挑行人最多的一帧
    frame = max(labels, key=lambda f: len(labels[f]))
    k0, k1 = video_index(frame, fps, 0), video_index(frame, fps, offset)
    imgs, _ = read_frames(path, {k0, k1})
    boxes = [b for _, b, _ in labels[frame]]
    left = draw_boxes(imgs[k0].copy(), boxes, (0, 255, 0))
    right = draw_boxes(imgs[k1].copy(), boxes, (0, 255, 0))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, img, title in [(axes[0], left, "offset 0 (by timestamp)"),
                           (axes[1], right, f"offset {offset} (found by detector IoU)")]:
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{OUT}/align.png", dpi=120)
    print("figures/align.png", f"偏移 {offset}，分数 {score:.2f}")


def fig_gif(video=GIF_VIDEO, start=GIF_START, n=50):
    """卡尔曼跟踪结果的动图：每个编号一个固定颜色。"""
    dets = load_dets(f"dets/{video}.txt")
    tracker = Tracker(Kalman())
    rng = np.random.default_rng(0)
    colors = {}
    frames = []
    # 跟踪器从第 1 帧就开始跑（这样编号是“真实”的），只是从 start 帧开始存图
    for frame in range(1, start + n):
        out = tracker.step(dets.get(frame, []))
        if frame < start:
            continue
        img = cv2.imread(f"{DATA}/seqs/{video}/img1/{frame:06d}.jpg")
        for tid, box in out:
            if tid not in colors:
                colors[tid] = tuple(int(c) for c in rng.integers(60, 255, 3))
            draw_boxes(img, [box], colors[tid], [tid], font=0.9)
        # 只留中间一条（行人都在这里），去掉天空和车头，再缩小
        img = cv2.resize(img[60:420], (800, 225))
        # 动图每帧最多 256 色，先压到 128 色让文件小一点
        frames.append(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).quantize(128))
    # 5 帧/秒，每帧 200 毫秒
    frames[0].save(f"{OUT}/tracking.gif", save_all=True, append_images=frames[1:],
                   duration=200, loop=0)
    print("figures/tracking.gif")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    figs = {"compare": fig_compare, "frame_iou": fig_frame_iou, "align": fig_align, "gif": fig_gif}
    names = sys.argv[1:] or list(figs)
    for name in names:
        figs[name]()
