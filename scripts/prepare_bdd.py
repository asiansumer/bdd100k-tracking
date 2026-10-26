"""第 5 关：从 .mov 视频里抽出和标注对应的帧，并写好 MOT17 格式的标注。

为什么要“对齐”：Kaggle 的视频是每秒 30 帧，标注是每秒 5 帧（大约每 6 帧视频取 1 帧）。
按时间算，第 i 个标注帧应该在视频的第 i × 30 / 5 帧，但实际上整体差了几帧。
所以先用检测器试几个偏移量，看哪个偏移下“检测框和标注框最对得上”，再按它抽帧。
"""
import os

import cv2
from ultralytics import YOLO

from bdd_io import read_bdd_parquet, load_ped_labels, write_mot_gt, write_seqinfo
from box_utils import iou

VIDEOS = [
    "00abd8a7-ecd6fc56", "000d35d3-41990aa4", "00699de6-58847872", "002d290d-90f2bab2",
    "007aeb45-c601742b", "01a01af7-76f6a476", "0204aad7-8ad8ba92", "02645e20-f0955341",
    "000f157f-dab3a407", "021cb34f-455d80a0",
]
DATA = "data/bdd_mot"
MODEL_PATH = "../yolov8-bdd100k-pedestrian/runs/yolov8m_bdd100k_pedestrian.pt"
# 要试的偏移量：-18 到 +6 帧
OFFSETS = range(-18, 7)


def video_index(frame, fps, offset):
    """第 frame 个标注帧（从 1 开始）对应视频里的第几帧（从 0 开始）。"""
    k = round((frame - 1) * fps / 5) + offset
    # 偏移是负数时，开头几帧会算出负数，就用视频第 0 帧
    return max(k, 0)


def read_frames(video_path, wanted):
    """从头到尾读一遍视频，只留下 wanted 里这些帧号的画面，返回 {帧号: 图片}。"""
    cap = cv2.VideoCapture(video_path)
    out = {}
    k = 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if k in wanted:
            out[k] = img
        k += 1
    cap.release()
    return out, k


def match_score(gts, dets):
    """每个标注框找 IoU 最大的检测框，返回这些最大 IoU 的平均值。越大说明越对得上。"""
    if not gts:
        return 0.0
    total = 0.0
    for _, g, _ in gts:
        best = 0.0
        for d in dets:
            best = max(best, iou(g, d))
        total += best
    return total / len(gts)


def find_offset(video_path, labels, fps, model):
    """挑 3 个行人比较多的标注帧，把每个偏移量都试一遍，返回平均分最高的偏移量。"""
    frames = sorted(labels)
    # 在开头、中间、结尾附近各挑一帧
    sample = [frames[len(frames) // 6], frames[len(frames) // 2], frames[len(frames) * 5 // 6]]
    wanted = set()
    for f in sample:
        for off in OFFSETS:
            wanted.add(video_index(f, fps, off))
    imgs, _ = read_frames(video_path, wanted)

    best_off, best_score = 0, -1.0
    for off in OFFSETS:
        score = 0.0
        for f in sample:
            img = imgs.get(video_index(f, fps, off))
            if img is None:
                continue
            res = model(img, conf=0.3, verbose=False)[0]
            score += match_score(labels[f], res.boxes.xyxy.tolist())
        score /= len(sample)
        if score > best_score:
            best_off, best_score = off, score
    return best_off, best_score


def prepare_video(video, df, model):
    video_path = f"{DATA}/videos/{video}.mov"
    labels = load_ped_labels(df, video)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    offset, score = find_offset(video_path, labels, fps, model)
    print(f"{video}: 偏移 {offset} 帧，对齐分数 {score:.2f}")

    # 和 MOT17 一样的目录结构：seqs/视频名/img1/000001.jpg、gt/gt.txt、seqinfo.ini
    seq_dir = f"{DATA}/seqs/{video}"
    os.makedirs(f"{seq_dir}/img1", exist_ok=True)
    os.makedirs(f"{seq_dir}/gt", exist_ok=True)

    # 每个标注帧对应的视频帧号；超出视频长度的就用最后一帧
    need = {}
    for f in labels:
        need[f] = video_index(f, fps, offset)
    imgs, n_video = read_frames(video_path, set(need.values()))
    last = max(imgs)
    for f, k in need.items():
        img = imgs.get(k, imgs[last])
        cv2.imwrite(f"{seq_dir}/img1/{f:06d}.jpg", img)

    write_mot_gt(labels, f"{seq_dir}/gt/gt.txt")
    write_seqinfo(f"{seq_dir}/seqinfo.ini", video, len(labels), width, height)
    n_ped = len({tid for boxes in labels.values() for tid, _, _ in boxes})
    print(f"  {len(labels)} 帧，{n_ped} 个行人，视频共 {n_video} 帧")


if __name__ == "__main__":
    model = YOLO(MODEL_PATH)
    df = read_bdd_parquet(f"{DATA}/mot_labels.parquet")
    for v in VIDEOS:
        prepare_video(v, df, model)
