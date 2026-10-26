"""第 5 关：用 P1 训练的 YOLOv8m 行人检测器，把每段视频的每一帧都检测一遍。

结果存成 dets/视频名.txt，每行：帧号,x1,y1,x2,y2,分数（和 P2 的格式一样）。
conf=0.1 故意设得很低，把低分框也留下来，第 6 关 ByteTrack 式的二次匹配要用到它们。
"""
import os

from ultralytics import YOLO

from scripts.prepare_bdd import VIDEOS, DATA, MODEL_PATH


def detect_video(model, video, device):
    img_dir = f"{DATA}/seqs/{video}/img1"
    n_frames = len(os.listdir(img_dir))
    os.makedirs("dets", exist_ok=True)
    with open(f"dets/{video}.txt", "w") as f:
        for frame in range(1, n_frames + 1):
            res = model(f"{img_dir}/{frame:06d}.jpg", conf=0.1, device=device, verbose=False)[0]
            for box, score in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
                x1, y1, x2, y2 = box
                f.write(f"{frame},{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f},{score:.3f}\n")
    print(video, "检测完成，共", n_frames, "帧")


if __name__ == "__main__":
    import sys
    # Mac 上用 mps（苹果芯片的 GPU）；别的电脑可以传 cpu
    device = sys.argv[1] if len(sys.argv) > 1 else "mps"
    model = YOLO(MODEL_PATH)
    for v in VIDEOS:
        detect_video(model, v, device)
