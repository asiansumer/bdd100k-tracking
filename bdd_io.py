import pandas as pd


def read_bdd_parquet(path):
    """读 Kaggle 版 BDD100K 跟踪标注，把带点的列名改成好用的名字。"""
    df = pd.read_parquet(path)
    return df.rename(columns={
        "box2d.x1": "x1", "box2d.y1": "y1", "box2d.x2": "x2", "box2d.y2": "y2",
        "attributes.crowd": "crowd",
    })


def load_ped_labels(df, video):
    """取出一段视频的行人标注。

    返回 {帧号: [(编号, (x1, y1, x2, y2), 是否人群框), ...]}，帧号从 1 开始（和 MOT 格式一致）。
    人群框：一群挤在一起的人只画了一个大框，BDD 官方评估时这种框不算分。
    """
    vdf = df[df["videoName"] == video]
    # BDD 的 frameIndex 从 0 开始，这段视频一共有多少个标注帧
    n_frames = int(vdf["frameIndex"].max()) + 1
    # 先给每一帧放一个空列表，没有行人的帧也要有
    labels = {f: [] for f in range(1, n_frames + 1)}
    peds = vdf[vdf["category"] == "pedestrian"]
    for row in peds.itertuples(index=False):
        # frameIndex 从 0 开始，MOT 格式的帧号从 1 开始
        frame = row.frameIndex + 1
        labels[frame].append((int(row.id), (row.x1, row.y1, row.x2, row.y2), bool(row.crowd)))
    return labels


def write_mot_gt(labels, path):
    """把标注写成 MOT17 的 gt.txt：帧号,编号,左上x,左上y,宽,高,1,类别,1

    类别 1 = 行人；类别 8 = MOT17 里的“干扰物”，TrackEval 会把落在它上面的输出框忽略掉，
    正好用来表示人群框“不算分”。
    """
    with open(path, "w") as f:
        for frame in sorted(labels):
            for tid, (x1, y1, x2, y2), crowd in labels[frame]:
                cls = 8 if crowd else 1
                f.write(f"{frame},{tid},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f},1,{cls},1\n")


def write_seqinfo(path, name, n_frames, width, height, fps=5):
    """TrackEval 需要的序列信息文件，格式和 MOT17 一样。"""
    with open(path, "w") as f:
        f.write("[Sequence]\n")
        f.write(f"name={name}\n")
        f.write("imDir=img1\n")
        f.write(f"frameRate={fps}\n")
        f.write(f"seqLength={n_frames}\n")
        f.write(f"imWidth={width}\n")
        f.write(f"imHeight={height}\n")
        f.write("imExt=.jpg\n")


def load_dets(path):
    """读检测结果文件，返回 {帧号: [((x1, y1, x2, y2), 分数), ...]}。"""
    dets = {}
    with open(path) as f:
        for line in f:
            frame, x1, y1, x2, y2, score = line.strip().split(",")
            dets.setdefault(int(frame), []).append(
                ((float(x1), float(y1), float(x2), float(y2)), float(score)))
    return dets
