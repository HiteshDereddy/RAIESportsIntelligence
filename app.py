import streamlit as st
import os
import json
import tempfile

import torch
import torch.nn as nn
import pandas as pd
import cv2
import numpy as np
from ultralytics import YOLO
from scipy.interpolate import interp1d

# ============================================================
# DEVICE
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# MODEL
# ============================================================
class TrajectoryTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=8, num_encoder_layers=1,
                 num_decoder_layers=1, dim_feedforward=2048, max_len=5000, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(2, d_model)
        self.pos_encoder = nn.Embedding(max_len, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        dec = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(enc, num_encoder_layers)
        self.decoder = nn.TransformerDecoder(dec, num_decoder_layers)

        self.pred_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)
        )

    def forward(self, src, tgt, src_pos, tgt_pos):
        src = self.layer_norm(self.input_proj(src) + self.pos_encoder(src_pos))
        memory = self.encoder(src)
        tgt = self.input_proj(tgt) + self.pos_encoder(tgt_pos)
        out = self.decoder(tgt, memory)
        return self.pred_head(out)


# ============================================================
# UTILS
# ============================================================
def rollout(model, coords, steps=5):
    coords = coords.unsqueeze(0)
    src = coords[:, :-steps]
    src_pos = torch.arange(src.size(1)).unsqueeze(0).to(device)
    memory = model.encoder(model.input_proj(src) + model.pos_encoder(src_pos))
    tgt = coords[:, -steps:-steps+1]

    preds = []
    for _ in range(steps):
        tgt_pos = torch.arange(tgt.size(1)).unsqueeze(0).to(device)
        out = model.decoder(model.input_proj(tgt) + model.pos_encoder(tgt_pos), memory)
        next_xy = model.pred_head(out[:, -1:])
        preds.append(next_xy)
        tgt = torch.cat([tgt, next_xy], dim=1)

    return torch.cat(preds, dim=1).squeeze(0)


def interpolate_trajectory(frame_ids, coords, target_ids):
    fx = interp1d(frame_ids, np.array(coords)[:, 0], fill_value="extrapolate")
    fy = interp1d(frame_ids, np.array(coords)[:, 1], fill_value="extrapolate")
    return target_ids, np.vstack([fx(target_ids), fy(target_ids)]).T.tolist()


# ============================================================
# MAIN PIPELINE
# ============================================================
def automated_trajectory_prediction(
    video_path,
    annotations_csv,
    yolo_model_path,
    trajectory_model_path,
    rollout_steps,
    min_trajectory_length,
    ball_class,
    conf_threshold,
    delivery_id
):
    annotations = pd.read_csv(annotations_csv).dropna(subset=["x", "y"])
    group = annotations.groupby("delivery_id").get_group(delivery_id)

    frames = sorted(group.frame_id.unique())
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    yolo = YOLO(yolo_model_path)
    yolo.overrides["conf"] = conf_threshold

    model = TrajectoryTransformer().to(device)
    model.load_state_dict(torch.load(trajectory_model_path, map_location=device))
    model.eval()

    traj = []
    for fid in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
    
        results = yolo.track(frame, persist=True, classes=[ball_class], verbose=False)
    
        if results[0].boxes is None or len(results[0].boxes) == 0:
            continue
    
        box = results[0].boxes[0]
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        traj.append((fid, (x1 + x2) / 2, (y1 + y2) / 2))
    
    # 🔥 CRITICAL SAFETY CHECK
    if len(traj) < min_trajectory_length:
        raise RuntimeError(
            f"No ball detected (traj length={len(traj)}). "
            f"Check YOLO model, ball_class, or conf_threshold."
        )

    f_ids, coords = zip(*[(f, (x, y)) for f, x, y in traj])
    full_ids, full_coords = interpolate_trajectory(f_ids, coords, frames)

    coords_t = torch.tensor(full_coords, dtype=torch.float32).to(device)
    preds = rollout(model, coords_t, rollout_steps)

    out = cv2.VideoWriter(
        f"delivery_{delivery_id}_predictions.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (1280, 720)
    )

    cap = cv2.VideoCapture(video_path)
    for i, fid in enumerate(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
        ret, frame = cap.read()
        if not ret:
            continue
        frame = cv2.resize(frame, (1280, 720))

        x, y = map(int, full_coords[i])
        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        if fid in frames[-rollout_steps:]:
            px, py = map(int, preds[frames[-rollout_steps:].index(fid)])
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)

        out.write(frame)

    out.release()


# ============================================================
# STREAMLIT UI
# ============================================================
st.set_page_config(page_title="Cricket Ball Trajectory", layout="centered")
st.title("Cricket Ball Trajectory Prediction")

video = st.file_uploader("Upload match video", type=["mp4", "avi", "webm"])
run = st.button("Run Prediction")

if run:
    if video is None:
        st.error("Upload a video")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video.read())
        video_path = tmp.name
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    automated_trajectory_prediction(
        video_path=video_path,
        annotations_csv=os.path.join(BASE_DIR, "data/smoothed_trajcomp.csv"),
        yolo_model_path=os.path.join(BASE_DIR, "models/best.pt"),
        trajectory_model_path=os.path.join(BASE_DIR, "models/aa-2.pth"),
        rollout_steps=5,
        min_trajectory_length=7,
        ball_class=0,
        conf_threshold=0.25,
        delivery_id=1
    )

    st.success("Done")
    st.video("delivery_1_predictions.mp4")
    st.subheader("⬇ Downloads")

    video_out = "delivery_1_predictions.mp4"
    csv_out = f"delivery_1_predictions.csv"
    json_out = f"delivery_1_predictions.json"
    
    # Create minimal CSV / JSON if you want them downloadable
    pd.DataFrame([{"delivery_id": 1}]).to_csv(csv_out, index=False)
    with open(json_out, "w") as f:
        json.dump({"delivery_id": 1}, f)
    
    with open(video_out, "rb") as f:
        st.download_button(
            label="Download Video",
            data=f,
            file_name=video_out,
            mime="video/mp4"
        )
    
    with open(csv_out, "rb") as f:
        st.download_button(
            label="Download CSV",
            data=f,
            file_name=csv_out,
            mime="text/csv"
        )
    
    with open(json_out, "rb") as f:
        st.download_button(
            label="Download JSON",
            data=f,
            file_name=json_out,
            mime="application/json"
        )

