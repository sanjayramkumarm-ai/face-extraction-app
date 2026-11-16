import os
import io
import cv2
import zipfile
import tempfile
import numpy as np
import streamlit as st

st.set_page_config(page_title="GodMode V2 Ultra Face Extractor", page_icon="🔥", layout="wide")

# ========================= Mediapipe =========================
try:
    import mediapipe as mp
    mp_face_mesh = mp.solutions.face_mesh
    MEDIAPIPE_OK = True
except Exception:
    MEDIAPIPE_OK = False

# ========================= Config =========================
TARGET_MIN = 10
TARGET_MAX = 15
OUT_SIZE = 320

# preferred quota (sum=15)
ANGLE_QUOTA = {
    "LEFT": 3,
    "RIGHT": 3,
    "CENTER": 3,
    "UP": 3,
    "DOWN": 3
}

# ========================= Utility =========================
def lap_var(img_bgr):
    g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(g, cv2.CV_64F).var()

def frame_hash(frame):
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(g, (24, 24), interpolation=cv2.INTER_AREA)
    return hash(small.tobytes())

def safe_crop(img, x1, y1, x2, y2):
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return img[y1:y2, x1:x2]

def make_zip(paths):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for p in paths:
            z.write(p, arcname=os.path.basename(p))
    buf.seek(0)
    return buf

# ========================= Landmark Geometry =========================
# MediaPipe indices
IDX_NOSE = 1
IDX_LEFT_CHEEK = 234
IDX_RIGHT_CHEEK = 454
IDX_FOREHEAD = 10
IDX_CHIN = 152

# eyes (rough stable points)
L_EYE_OUTER = 33
L_EYE_INNER = 133
L_EYE_UP = 159
L_EYE_DOWN = 145

R_EYE_OUTER = 263
R_EYE_INNER = 362
R_EYE_UP = 386
R_EYE_DOWN = 374

# face oval for crop
FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109
]

def to_xy(landmarks, w, h):
    return np.array([(p.x * w, p.y * h) for p in landmarks], dtype=np.float32)

def estimate_pose(lm_xy):
    nose = lm_xy[IDX_NOSE]
    lc = lm_xy[IDX_LEFT_CHEEK]
    rc = lm_xy[IDX_RIGHT_CHEEK]
    fh = lm_xy[IDX_FOREHEAD]
    ch = lm_xy[IDX_CHIN]

    face_w = np.linalg.norm(rc - lc) + 1e-6
    face_h = np.linalg.norm(ch - fh) + 1e-6

    cheek_mid_x = (lc[0] + rc[0]) / 2.0
    yaw_norm = (nose[0] - cheek_mid_x) / face_w
    yaw = float(np.clip(yaw_norm * 125.0, -75, 75))

    mid_y = (fh[1] + ch[1]) / 2.0
    pitch_norm = (nose[1] - mid_y) / face_h
    pitch = float(np.clip(pitch_norm * 140.0, -60, 60))
    return yaw, pitch

def angle_bucket(yaw, pitch):
    # prioritize pitch
    if pitch < -16:
        return "UP"
    if pitch > 16:
        return "DOWN"
    if yaw < -14:
        return "LEFT"
    if yaw > 14:
        return "RIGHT"
    return "CENTER"

def eye_open_score(lm_xy):
    # EAR-like for both eyes
    l_h = np.linalg.norm(lm_xy[L_EYE_INNER] - lm_xy[L_EYE_OUTER]) + 1e-6
    l_v = np.linalg.norm(lm_xy[L_EYE_UP] - lm_xy[L_EYE_DOWN])
    r_h = np.linalg.norm(lm_xy[R_EYE_INNER] - lm_xy[R_EYE_OUTER]) + 1e-6
    r_v = np.linalg.norm(lm_xy[R_EYE_UP] - lm_xy[R_EYE_DOWN])

    l_ratio = l_v / l_h
    r_ratio = r_v / r_h
    return float((l_ratio + r_ratio) / 2.0)

def eye_state_from_score(score):
    # tune if needed for your camera
    return "OPEN" if score > 0.19 else "CLOSED"

# ========================= Crop =========================
def crop_face_from_landmarks(frame, lm_xy, angle):
    h, w = frame.shape[:2]
    pts = lm_xy[FACE_OVAL].astype(np.int32)
    x, y, bw, bh = cv2.boundingRect(pts)
    if bw < 45 or bh < 45:
        return None

    # adaptive padding by angle
    if angle == "UP":
        p_top, p_bottom, p_side = 0.34, 0.12, 0.20
    elif angle == "DOWN":
        p_top, p_bottom, p_side = 0.14, 0.34, 0.20
    else:
        p_top, p_bottom, p_side = 0.24, 0.18, 0.20

    x1 = int(x - bw * p_side)
    x2 = int(x + bw + bw * p_side)
    y1 = int(y - bh * p_top)
    y2 = int(y + bh + bh * p_bottom)

    crop = safe_crop(frame, x1, y1, x2, y2)
    if crop is None:
        return None

    ch, cw = crop.shape[:2]
    if ch < 90 or cw < 90:
        return None

    crop = cv2.resize(crop, (OUT_SIZE, OUT_SIZE), interpolation=cv2.INTER_CUBIC)
    return crop

# ========================= Quality + Dedup =========================
def quality_score(face):
    g = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    sharp = lap_var(face)
    mean = float(np.mean(g))
    std = float(np.std(g))

    score = 0.0
    score += min(sharp / 220.0, 1.0) * 45.0

    if 50 <= mean <= 205:
        score += 25.0
    elif 35 <= mean <= 225:
        score += 12.0

    score += min(std / 60.0, 1.0) * 20.0

    c = g[OUT_SIZE//4:3*OUT_SIZE//4, OUT_SIZE//4:3*OUT_SIZE//4]
    cstd = float(np.std(c))
    score += min(cstd / 45.0, 1.0) * 10.0

    return float(np.clip(score, 0, 100))

def hist_feature(face):
    hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0,1], None, [32,32], [0,180,0,256])
    h = cv2.normalize(h, h).flatten()
    return h

def is_duplicate(hist_list, new_hist, threshold=0.92):
    # correlation near 1 => very similar
    for h in hist_list:
        c = cv2.compareHist(h.astype(np.float32), new_hist.astype(np.float32), cv2.HISTCMP_CORREL)
        if c >= threshold:
            return True
    return False

# ========================= Selection =========================
def select_ultra(cands, target_max=15):
    # First take by angle quota
    by_angle = {k: [] for k in ANGLE_QUOTA.keys()}
    for i, c in enumerate(cands):
        by_angle[c["angle"]].append((i, c["score"]))

    for k in by_angle:
        by_angle[k].sort(key=lambda x: x[1], reverse=True)

    selected = []
    selected_set = set()

    for k, q in ANGLE_QUOTA.items():
        taken = 0
        for idx, _ in by_angle[k]:
            if idx not in selected_set:
                selected.append(idx)
                selected_set.add(idx)
                taken += 1
            if taken >= q:
                break

    # Fill remaining by quality
    if len(selected) < target_max:
        all_idx = sorted(range(len(cands)), key=lambda i: cands[i]["score"], reverse=True)
        for i in all_idx:
            if i not in selected_set:
                selected.append(i)
                selected_set.add(i)
            if len(selected) >= target_max:
                break

    # If too many
    selected = selected[:target_max]

    # Ensure at least TARGET_MIN if possible
    if len(selected) < TARGET_MIN:
        all_idx = sorted(range(len(cands)), key=lambda i: cands[i]["score"], reverse=True)
        for i in all_idx:
            if i not in selected_set:
                selected.append(i)
                selected_set.add(i)
            if len(selected) >= min(TARGET_MIN, len(cands)):
                break

    return [cands[i] for i in selected]

# ========================= Pipeline =========================
def process_video(uploaded, progress, status):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        video_path = tmp.name

    candidates = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            status.error("Cannot open video.")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration = total / fps if fps > 0 else 0

        if duration > 120:
            status.error("Video too long (>120s).")
            return []

        sample_step = max(1, int(fps * 0.10))
        seen_frame_hash = set()
        seen_face_hist = []

        mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.55
        )

        idx = 0
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            if idx % sample_step != 0:
                idx += 1
                continue

            if lap_var(frame) < 35:
                idx += 1
                continue

            fh = frame_hash(frame)
            if fh in seen_frame_hash:
                idx += 1
                continue
            seen_frame_hash.add(fh)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mesh.process(rgb)

            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                h, w = frame.shape[:2]
                lm_xy = to_xy(lm, w, h)

                yaw, pitch = estimate_pose(lm_xy)
                angle = angle_bucket(yaw, pitch)

                eye_score = eye_open_score(lm_xy)
                eye_state = eye_state_from_score(eye_score)

                face = crop_face_from_landmarks(frame, lm_xy, angle)
                if face is None:
                    idx += 1
                    continue

                q = quality_score(face)
                if q < 52:
                    idx += 1
                    continue

                feat = hist_feature(face)
                if is_duplicate(seen_face_hist, feat, threshold=0.92):
                    idx += 1
                    continue

                seen_face_hist.append(feat)

                candidates.append({
                    "image": face,
                    "score": q,
                    "yaw": yaw,
                    "pitch": pitch,
                    "angle": angle,
                    "eye_score": eye_score,
                    "eye_state": eye_state,
                    "frame_idx": idx
                })

            idx += 1
            if total > 0:
                progress.progress(min(idx / total, 0.97))
            status.info(f"Candidates: {len(candidates)}")

        cap.release()
        mesh.close()

    finally:
        try:
            os.unlink(video_path)
        except Exception:
            pass

    if not candidates:
        return []

    selected = select_ultra(candidates, target_max=TARGET_MAX)
    return selected

# ========================= Save + UI =========================
def save_faces(faces, out_dir="output"):
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(out_dir):
        if f.lower().endswith(".jpg"):
            try:
                os.remove(os.path.join(out_dir, f))
            except Exception:
                pass

    paths = []
    for i, item in enumerate(faces, 1):
        p = os.path.join(out_dir, f"face_{i:02d}.jpg")
        cv2.imwrite(p, item["image"], [cv2.IMWRITE_JPEG_QUALITY, 97])
        paths.append(p)
    return paths

def main():
    st.title("🔥 GodMode V2 Ultra Face Extractor")
    st.write("Pose + eye-state + anti-duplicate + quota diversity (10–15 faces)")

    if not MEDIAPIPE_OK:
        st.error("Install dependency first: `pip install mediapipe`")
        return

    file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"])

    if file and st.button("Extract Ultra Faces", type="primary", use_container_width=True):
        prog = st.progress(0)
        stat = st.empty()

        faces = process_video(file, prog, stat)
        if not faces:
            stat.error("No valid faces found. Try brighter video + more head movement.")
            return

        paths = save_faces(faces)
        prog.progress(1.0)
        stat.success(f"Done ✅ Extracted {len(paths)} faces.")

        # Metrics
        angle_counts = {"LEFT":0, "RIGHT":0, "CENTER":0, "UP":0, "DOWN":0}
        eye_open = 0
        eye_closed = 0
        for f in faces:
            angle_counts[f["angle"]] += 1
            if f["eye_state"] == "OPEN":
                eye_open += 1
            else:
                eye_closed += 1

        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("LEFT", angle_counts["LEFT"])
        c2.metric("RIGHT", angle_counts["RIGHT"])
        c3.metric("CENTER", angle_counts["CENTER"])
        c4.metric("UP", angle_counts["UP"])
        c5.metric("DOWN", angle_counts["DOWN"])

        e1,e2,e3 = st.columns(3)
        e1.metric("Eyes Open", eye_open)
        e2.metric("Eyes Closed", eye_closed)
        e3.metric("Total", len(paths))

        st.subheader("Extracted Faces")
        cols = st.columns(5)
        for i, p in enumerate(paths):
            img = cv2.imread(p)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            meta = faces[i]
            cap = f"{os.path.basename(p)} | {meta['angle']} | {meta['eye_state']}"
            with cols[i % 5]:
                # FIXED: Removed use_container_width parameter
                st.image(img, caption=cap, width=300)

        st.download_button(
            "📦 Download ZIP",
            data=make_zip(paths),
            file_name="ultra_faces.zip",
            mime="application/zip",
            use_container_width=True
        )

if __name__ == "__main__":
    main()