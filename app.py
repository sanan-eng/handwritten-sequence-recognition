# app.py
import streamlit as st
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Device ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- Model classes ----------------
class CNNBackbone(nn.Module):
    def __init__(self, input_height=32):
        super().__init__()
        conv_output_height = input_height // 2 // 2
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, 1, 1, groups=128), nn.Conv2d(128, 256, 1, 1), nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.adaptive_pool = nn.AdaptiveAvgPool2d((conv_output_height, None))

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.adaptive_pool(x)
        x = x.permute(0, 3, 1, 2)
        b, seq_len = x.size(0), x.size(1)
        x = x.contiguous().view(b, seq_len, -1)
        return x


class CRNN(nn.Module):
    def __init__(self, input_height=32, num_classes=11, hidden_size=128, num_layers=2, dropout=0.3):
        super().__init__()
        self.cnn = CNNBackbone(input_height)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, input_height, 100)
            rnn_input = self.cnn(dummy)
        self.rnn = nn.LSTM(rnn_input.size(2), hidden_size, num_layers=num_layers, dropout=dropout,
                           bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.cnn(x)
        x = self.dropout(x)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = self.fc(x)
        return F.log_softmax(x, dim=2)


# ---------------- Decoding ----------------
BLANK_INDEX = 10


def greedy_ctc_decode(log_probs, blank_index=BLANK_INDEX, collapse_repeats=True,
                      confidence=None, conf_threshold=0.5, min_repeat_confidence=0.7):
    """
    Improved CTC greedy decoder that:
    - Collapses repeats when appropriate
    - Preserves repeated characters if their confidence is high
    - Uses per-character confidence to prevent losing repeated digits
    """
    _, max_idx = torch.max(log_probs, dim=2)
    probs = torch.exp(log_probs)
    batch_size, seq_len, num_classes = log_probs.shape

    decoded, char_conf = [], []

    for b in range(batch_size):
        seq = max_idx[b].cpu().numpy().tolist()
        # Get confidence scores for all characters
        confs = confidence[b].cpu().numpy().tolist() if confidence is not None else [0.0] * seq_len

        out_seq, out_conf = [], []
        prev_char = blank_index
        prev_char_pos = -1

        for t, char in enumerate(seq):
            char_prob = probs[b, t, char].item()

            if char == blank_index:
                continue

            if collapse_repeats and char == prev_char:
                # Check if we should preserve the repeated character
                # Based on confidence and distance from previous occurrence
                time_since_prev = t - prev_char_pos
                avg_confidence = (confs[t] + confs[prev_char_pos]) / 2

                # Preserve repeat if confidence is high or it's spaced out
                if (avg_confidence >= min_repeat_confidence or
                        time_since_prev > 5 or
                        confs[t] >= conf_threshold):
                    out_seq.append(str(char))
                    out_conf.append(confs[t])
                    prev_char_pos = t
            else:
                # Always add non-repeated characters
                out_seq.append(str(char))
                out_conf.append(confs[t])
                prev_char = char
                prev_char_pos = t

        decoded.append("".join(out_seq))
        char_conf.append(out_conf)

    return decoded, char_conf


def beam_search_decode(log_probs, beam_width=5, blank_index=BLANK_INDEX):
    preds = log_probs.cpu().numpy()
    batch_size, seq_len, num_classes = preds.shape
    results = []

    for b in range(batch_size):
        beams = [('', 1.0, 0)]  # (sequence, score, last_char)

        for t in range(seq_len):
            probs = np.exp(preds[b, t])
            new_beams = []

            for seq, score, last_char in beams:
                for c in range(num_classes):
                    p = probs[c]
                    new_score = score * p

                    if c == blank_index:
                        # Keep the sequence unchanged
                        new_beams.append((seq, new_score, last_char))
                    elif c == last_char:
                        # Repeated character - keep it with penalty
                        penalty = 0.7  # Penalty for repeats
                        new_seq = seq + str(c)
                        new_beams.append((new_seq, new_score * penalty, c))
                    else:
                        # New character
                        new_seq = seq + str(c)
                        new_beams.append((new_seq, new_score, c))

            # Prune to keep only top beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]

        # Return the best sequence
        best_seq = beams[0][0] if beams else ''
        results.append(best_seq)

    return results


# ---------------- Preprocessing ----------------
IMG_H, IMG_W = 32, 160
GLOBAL_MEAN, GLOBAL_STD = 0.1307, 0.3081


def auto_invert(img):
    gray = img.convert("L")
    if np.array(gray).mean() > 127: return ImageOps.invert(gray)
    return gray


def denoise(img, radius=0):
    return img if radius <= 0 else img.filter(ImageFilter.MedianFilter(radius))


def preprocess(img, resize_h=IMG_H, resize_w=IMG_W, normalize=True, keep_aspect=True):
    orig = img.convert("RGB")
    gray = auto_invert(orig)
    w, h = gray.size
    if keep_aspect:
        scale = resize_h / h
        new_w = int(w * scale)
        resized = gray.resize((min(new_w, resize_w), resize_h))
        pad_right = resize_w - resized.size[0]
        resized = ImageOps.expand(resized, border=(0, 0, pad_right, 0), fill=0)
    else:
        resized = gray.resize((resize_w, resize_h))
    tensor = transforms.ToTensor()(resized)
    if normalize:
        tensor = transforms.Normalize((GLOBAL_MEAN,), (GLOBAL_STD,))(tensor)
    return {"orig": orig, "gray": gray, "resized": resized, "tensor": tensor.unsqueeze(0).to(device)}


# ---------------- Model loader ----------------
CHECKPOINT_PATH = "checkpoints/best_model.pth"


@st.cache_resource(show_spinner=False)
def load_model(path=CHECKPOINT_PATH):
    if not os.path.exists(path): raise FileNotFoundError(f"{path} not found")
    checkpoint = torch.load(path, map_location=device)
    model = CRNN(input_height=IMG_H, num_classes=11)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device).eval()
    return model


def run_model(model, tensor):
    with torch.no_grad():
        log_probs = model(tensor)
    probs = torch.exp(log_probs)
    max_conf, _ = torch.max(probs, dim=2)
    return log_probs.cpu(), max_conf.cpu()


# ---------------- Visualization ----------------
def plot_heatmap(log_probs):
    probs = log_probs.exp().numpy()
    fig, ax = plt.subplots(figsize=(12, 4))
    sns.heatmap(probs.T, cmap="viridis", cbar=True, xticklabels=5, yticklabels=[str(i) for i in range(10)] + ["blank"],
                ax=ax)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Classes")
    return fig


def topk_table(log_probs, topk=3):
    T, C = log_probs.shape
    rows = []
    for t in range(T):
        p = log_probs[t].exp().numpy()
        idxs = p.argsort()[-topk:][::-1]
        rows.append((t, [(str(i) if i != BLANK_INDEX else "blank", float(p[i])) for i in idxs]))
    return rows


# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="Handwritten Seq Recognizer", layout="wide")
st.title("✍ Handwritten Digit Sequence Recognition — Advanced")

left, right = st.columns((1, 2))

with left:
    st.header("Input")
    uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    st.markdown("### Canvas Input")
    st.caption('Draw digits on canvas, process them, and download your handwritten input as a PNG for predicting with the recognition model')
    from streamlit_drawable_canvas import st_canvas

    canvas_result = st_canvas(
        fill_color="black",
        stroke_width=15,
        stroke_color="white",
        background_color="black",
        height=IMG_H * 4,
        width=IMG_W * 4,
        drawing_mode="freedraw",
        key="canvas",
    )
    if canvas_result.image_data is not None:
        arr = canvas_result.image_data.astype(np.uint8)
        img = Image.fromarray(arr).convert("RGB")

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        byte_im = buf.getvalue()

        st.download_button(
            label="💾 Save Canvas as PNG",
            data=byte_im,
            file_name="canvas_digit.png",
            mime="image/png"
        )
    st.markdown("### Preprocessing options")
    contrast = st.slider("Contrast", 1.0, 3.0, 1.6, 0.1)
    denoise_radius = st.slider("Denoise", 0, 5, 0)
    keep_aspect = st.checkbox("Keep aspect ratio & pad", True)

    st.markdown("### Decoding options")
    collapse_repeats = st.checkbox("Collapse repeats", True)
    conf_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
    min_repeat_confidence = st.slider("Min repeat confidence", 0.0, 1.0, 0.7, 0.05,
                                      help="Minimum confidence to preserve repeated characters")
    beam_search_toggle = st.checkbox("Beam-search fallback", True)
    beam_width = st.slider("Beam width", 2, 12, 5)
    topk = st.slider("Top-k per timestep", 1, 5, 3)

with right:
    st.header("Model & Debug")
    if not os.path.exists(CHECKPOINT_PATH):
        st.error("Checkpoint missing! Place best_model.pth")
        st.stop()
    model = load_model()
    st.write(f"Using device: {device}")
    if st.button("Reload model"):
        model = load_model()
        st.success("Model reloaded")

# ---------------- Processing ----------------
img_to_use = None
if uploaded:
    try:
        img_to_use = Image.open(uploaded)
    except:
        st.error("Could not open image"); st.stop()
elif canvas_result.image_data is not None:
    arr = canvas_result.image_data.astype(np.uint8)
    img_to_use = Image.fromarray(arr).convert("L")

if img_to_use:
    img = img_to_use.convert("RGB")  # convert uploaded images to RGB
    img = ImageOps.autocontrast(img)
    img = ImageEnhance.Contrast(img).enhance(contrast)
    if denoise_radius > 0: img = denoise(img, denoise_radius)
    proc = preprocess(img, keep_aspect=keep_aspect)
    tensor = proc["tensor"]

    st.subheader("Preprocessing preview")
    col1, col2, col3 = st.columns(3)
    col1.image(proc["orig"], "Original", use_container_width=True)
    col2.image(proc["gray"], "Gray", use_container_width=True)
    col3.image(proc["resized"], f"Resized ({IMG_H}x{IMG_W})", use_container_width=True)

    st.subheader("Model inference")
    log_probs, max_conf = run_model(model, tensor)
    st.write(f"Log-probs shape: {log_probs.shape}, Max conf shape: {max_conf.shape}")

    # Greedy decoding with improved algorithm
    seqs, confs = greedy_ctc_decode(log_probs, collapse_repeats=collapse_repeats,
                                    confidence=max_conf, conf_threshold=conf_threshold,
                                    min_repeat_confidence=min_repeat_confidence)
    greedy_seq = seqs[0]
    greedy_conf_list = confs[0]

    seq_conf = float(np.mean(greedy_conf_list)) if greedy_conf_list else float(max_conf.mean())
    st.success(f"Greedy Prediction: {greedy_seq or '---'}  (avg conf {seq_conf:.3f})")

    beam_seq = None
    if beam_search_toggle and (seq_conf < conf_threshold or not greedy_seq):
        beam_seq = beam_search_decode(log_probs, beam_width=beam_width)[0]
        st.info(f"Beam-search fallback: {beam_seq}")

    # Per-character confidences
    st.subheader("Character confidences")
    if greedy_seq:
        table_rows = [{"pos": i + 1, "char": ch, "conf": f"{greedy_conf_list[i]:.3f}"} for i, ch in
                      enumerate(greedy_seq)]
        st.table(table_rows)

    # Heatmap
    st.subheader("Probability heatmap")
    fig = plot_heatmap(log_probs[0])
    st.pyplot(fig)

    # Top-k
    st.subheader("Top-k per timestep")
    rows = topk_table(log_probs[0], topk=topk)
    for t, items in rows:
        st.write(f"t={t}: " + ", ".join([f"{c}:{p:.3f}" for c, p in items]))

    # Debug
    with st.expander("Debug info"):
        st.write("Log-probs (first 5 timesteps):", log_probs[0][:5].numpy())
        st.write("Max conf (first 20 timesteps):", max_conf[0][:20].numpy())
        st.write("Greedy decoded seq & confidences:", greedy_seq, greedy_conf_list)
        if beam_seq: st.write("Beam seq:", beam_seq)
else:
    st.info("Upload image or draw on canvas to predict digits.")
