#!/bin/bash
# 第 7 关：用 TrackEval 评估三种运动模型（和 P2 一样，在 moteval 环境里跑）
# 运行：conda activate moteval，然后在仓库根目录 bash scripts/evaluate.sh
# 结果：屏幕上的表格，和 results/<模型名>/pedestrian_summary.txt
SEQS=$(ls data/bdd_mot/seqs)
python ../TrackEval/scripts/run_mot_challenge.py \
    --BENCHMARK BDD \
    --GT_FOLDER data/bdd_mot/seqs \
    --TRACKERS_FOLDER results \
    --SKIP_SPLIT_FOL True \
    --TRACKERS_TO_EVAL no_motion alpha_beta kalman \
    --SEQ_INFO $SEQS \
    --METRICS HOTA CLEAR Identity \
    --USE_PARALLEL False \
    --PLOT_CURVES False
