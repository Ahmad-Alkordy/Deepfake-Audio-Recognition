import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
from pathlib import Path

st.set_page_config(page_title="VoiceShield", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=Inter:wght@300;400;500;600&display=swap');

:root {
    --bg:       #0d0f14;
    --surface:  #13161e;
    --surface2: #181c26;
    --border:   #252d3d;
    --border2:  #2e3a50;
    --accent:   #6c8fff;
    --accent2:  #a78bfa;
    --danger:   #f87171;
    --safe:     #34d399;
    --warn:     #fbbf24;
    --text:     #d4ddf0;
    --text2:    #8899b8;
    --text3:    #4a5a78;
    --mono:     'IBM Plex Mono', monospace;
    --sans:     'Inter', sans-serif;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: var(--sans) !important;
}
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem !important; max-width: 1080px !important; }
h1, h2, h3 { font-family: var(--sans) !important; color: #e8eeff !important; font-weight: 600 !important; }

.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-sm {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.2rem;
}
.pill {
    display: flex; flex-direction: column; align-items: center;
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 12px; padding: .9rem 1rem; text-align: center;
}
.pill .val {
    font-family: var(--mono); font-size: 1.35rem; font-weight: 600; color: var(--accent);
}
.pill .lbl {
    font-size: .68rem; color: var(--text3); text-transform: uppercase;
    letter-spacing: .1em; margin-top: .3rem;
}
.verdict-real {
    display: inline-flex; align-items: center; gap: .5rem;
    padding: .4rem 1.2rem; border-radius: 8px;
    background: rgba(52,211,153,.08); border: 1.5px solid var(--safe);
    color: var(--safe); font-family: var(--mono); font-size: 1rem; font-weight: 600;
}
.verdict-fake {
    display: inline-flex; align-items: center; gap: .5rem;
    padding: .4rem 1.2rem; border-radius: 8px;
    background: rgba(248,113,113,.08); border: 1.5px solid var(--danger);
    color: var(--danger); font-family: var(--mono); font-size: 1rem; font-weight: 600;
}
.bar-track {
    background: var(--surface2); border-radius: 6px; height: 8px;
    overflow: hidden; margin: .6rem 0 .3rem;
}
.bar-fill { height: 100%; border-radius: 6px; }
.stat-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: .35rem 0; border-bottom: 1px solid var(--border);
    font-size: .82rem;
}
.stat-row:last-child { border-bottom: none; }
.stat-label { color: var(--text2); }
.stat-val { font-family: var(--mono); color: var(--text); font-size: .8rem; }
.risk-box {
    margin-top: 1rem; padding: .7rem 1rem;
    border-radius: 8px; border-left: 3px solid;
    background: rgba(0,0,0,.25);
}
.risk-tag { font-family: var(--mono); font-weight: 600; font-size: .9rem; margin-top: .2rem; }
.risk-label { font-size: .68rem; color: var(--text3); text-transform: uppercase; letter-spacing: .08em; }

[data-testid="stFileUploader"] {
    border: 1.5px dashed var(--border2) !important;
    border-radius: 12px !important; background: var(--surface2) !important;
    transition: border-color .2s !important;
}
.stButton > button {
    background: var(--accent) !important; color: #fff !important;
    font-family: var(--sans) !important; font-weight: 600 !important;
    border: none !important; border-radius: 8px !important;
    padding: .55rem 1.4rem !important; transition: opacity .2s !important;
}
.stButton > button:hover { opacity: .85 !important; }
hr { border-color: var(--border) !important; margin: 1.4rem 0 !important; }
.stRadio label { font-family: var(--sans) !important; font-size: .9rem !important; }
.streamlit-expanderHeader {
    background: var(--surface2) !important; border: 1px solid var(--border) !important;
    border-radius: 8px !important;
}
div[data-testid="stMetricValue"] { font-family: var(--mono) !important; }

.step-num {
    flex-shrink: 0; width: 2.2rem; height: 2.2rem; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-family: var(--mono); font-size: .72rem; font-weight: 600;
    border: 2px solid;
}
.section-label {
    font-size: .68rem; font-weight: 600; letter-spacing: .12em;
    text-transform: uppercase; color: var(--text3); margin-bottom: .6rem;
}
</style>
""", unsafe_allow_html=True)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.shortcut = (
            nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False), nn.BatchNorm2d(out_ch))
            if stride != 1 or in_ch != out_ch else nn.Identity()
        )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(out)) + self.shortcut(x))


class DeepfakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem   = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(ResidualBlock(32, 64, 2),   ResidualBlock(64, 64))
        self.layer2 = nn.Sequential(ResidualBlock(64, 128, 2),  ResidualBlock(128, 128))
        self.layer3 = nn.Sequential(ResidualBlock(128, 256, 2), ResidualBlock(256, 256))
        self.pool   = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.4), nn.Linear(256, 128),
            nn.ReLU(inplace=True), nn.Dropout(0.3), nn.Linear(128, 1)
        )

    def forward(self, x):
        for layer in [self.stem, self.layer1, self.layer2, self.layer3, self.pool]:
            x = layer(x)
        return self.classifier(x)


TARGET_FRAMES = 87
N_MELS = 128
SR = 16_000
BG, GRID, TCOL = "#0a0d14", "#1e2840", "#6677a0"


def extract_mel(audio, sr):
    if sr != SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SR)
    mel = librosa.feature.melspectrogram(y=audio, sr=SR, n_mels=N_MELS)
    mel_db = librosa.power_to_db(mel)
    if mel_db.shape[1] < TARGET_FRAMES:
        mel_db = np.pad(mel_db, ((0, 0), (0, TARGET_FRAMES - mel_db.shape[1])))
    else:
        mel_db = mel_db[:, :TARGET_FRAMES]
    mel_db = (mel_db - mel_db.mean()) / (mel_db.std() + 1e-8)
    return mel_db.astype(np.float32)


@st.cache_resource
def load_model(path):
    m = DeepfakeCNN()
    m.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    m.eval()
    return m


def predict(model, mel, threshold=0.5):
    t = torch.tensor(mel).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        prob = torch.sigmoid(model(t).squeeze()).item()
    return ("FAKE" if prob >= threshold else "REAL"), prob


def style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=TCOL, labelsize=8)
    for s in ax.spines.values():
        s.set_edgecolor(GRID)
    ax.xaxis.label.set_color(TCOL)
    ax.yaxis.label.set_color(TCOL)
    ax.title.set_color("#c4d0e8")
    ax.grid(color=GRID, linewidth=0.4, alpha=0.6)


def plot_waveform(audio, sr):
    fig, ax = plt.subplots(figsize=(8, 1.7))
    fig.patch.set_facecolor(BG)
    t = np.linspace(0, len(audio) / sr, len(audio))
    ax.plot(t, audio, color="#6c8fff", linewidth=0.5, alpha=0.9)
    ax.fill_between(t, audio, alpha=0.12, color="#6c8fff")
    ax.set(xlabel="Time (s)", ylabel="Amplitude", title="Waveform")
    ax.title.set_size(9); ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8)
    style_ax(ax)
    fig.tight_layout(pad=0.4)
    return fig


def plot_spectrogram(mel):
    fig, ax = plt.subplots(figsize=(8, 2.3))
    fig.patch.set_facecolor(BG)
    cmap = mcolors.LinearSegmentedColormap.from_list("vs", ["#0a0d14", "#1a2a4a", "#6c8fff", "#a78bfa", "#f472b6"])
    img = ax.imshow(mel, origin="lower", aspect="auto", cmap=cmap,
                    extent=[0, TARGET_FRAMES, 0, N_MELS])
    cbar = fig.colorbar(img, ax=ax, pad=0.01)
    cbar.ax.tick_params(colors=TCOL, labelsize=7)
    cbar.set_label("dB (norm)", color=TCOL, fontsize=7)
    ax.set(xlabel="Time frames", ylabel="Mel bin", title="Mel Spectrogram")
    ax.title.set_size(9); ax.xaxis.label.set_size(8); ax.yaxis.label.set_size(8)
    style_ax(ax)
    fig.tight_layout(pad=0.4)
    return fig


# ── Sidebar ──
with st.sidebar:
    st.markdown("""
    <div style='padding:.4rem 0 1.4rem'>
      <div style='font-size:1.05rem;font-weight:700;color:#6c8fff;letter-spacing:.02em'>
        🛡️ VoiceShield
      </div>
      <div style='font-size:.72rem;color:#4a5a78;margin-top:.3rem;font-family:"IBM Plex Mono",monospace'>
        AUDIO FORENSICS v1.0
      </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio("nav", ["🔍  Analyze", "📊  Results", "📖  How It Works"],
                    label_visibility="collapsed")
    st.markdown("---")

    st.markdown("<div class='section-label'>Settings</div>", unsafe_allow_html=True)
    threshold = st.slider("Decision threshold", 0.1, 0.9, 0.5, 0.05,
                          help="Probability above which audio is classified FAKE")
    st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

    st.markdown("<div class='section-label'>Model weights</div>", unsafe_allow_html=True)
    model_file = st.file_uploader("Upload .pth file", type=["pth"], label_visibility="collapsed")

    st.markdown("---")
    # st.markdown("""
    # <div style='font-size:.72rem;color:#3a4a68;line-height:1.8'>
    #   <span style='color:#4a5a78'>Dataset</span><br>
    #   FOR · for-2sec subset<br>
    #   8,935 real + 8,935 fake<br><br>
    #   <span style='color:#4a5a78'>Test accuracy</span><br>
    #   <span style='font-family:"IBM Plex Mono",monospace;color:#6c8fff'>99.89%</span>
    #   &nbsp; ROC-AUC
    #   <span style='font-family:"IBM Plex Mono",monospace;color:#6c8fff'>100%</span>
    # </div>
    # """, unsafe_allow_html=True)

model = None
if model_file:
    tmp = Path("/tmp/model.pth")
    tmp.write_bytes(model_file.read())
    try:
        model = load_model(str(tmp))
    except Exception as e:
        st.sidebar.error(f"Load failed: {e}")


# ── Analyze ──
if "Analyze" in page:
    st.markdown("""
    <h1 style='font-size:1.7rem;margin:0'>Voice Authenticity Analyzer</h1>
    <p style='color:#6677a0;margin:.4rem 0 1.4rem;font-size:.9rem'>
      Upload a speech recording to check if it was AI-generated or voice-cloned
    </p>
    """, unsafe_allow_html=True)

    if not model:
        st.info("Upload model weights (.pth) in the sidebar to enable predictions. "
                "Visualizations work without a model.", icon="ℹ️")

    uploaded = st.file_uploader("Drop audio file here — WAV, MP3, FLAC, OGG, M4A",
                                type=["wav", "mp3", "flac", "ogg", "m4a"],
                                label_visibility="collapsed")

    if not uploaded:
        st.markdown("""
        <div style='text-align:center;padding:5rem 1rem'>
          <div style='font-size:3.5rem;opacity:.2'>🎙️</div>
          <div style='margin-top:1rem;color:#3a4a68;font-size:.9rem'>
            No file uploaded yet
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        raw = uploaded.read()
        try:
            audio, sr = librosa.load(io.BytesIO(raw), sr=None, mono=True)
        except Exception as e:
            st.error(f"Could not load audio: {e}")
            st.stop()

        dur = len(audio) / sr
        st.audio(raw, format=f"audio/{uploaded.name.split('.')[-1]}")
        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)

        for col, val, lbl in zip(
            st.columns(4),
            [f"{dur:.2f}s", f"{sr//1000} kHz", f"{len(raw)/1024:.0f} KB",
             uploaded.name.split(".")[-1].upper()],
            ["Duration", "Sample rate", "File size", "Format"]
        ):
            col.markdown(
                f"<div class='pill'><div class='val'>{val}</div><div class='lbl'>{lbl}</div></div>",
                unsafe_allow_html=True)

        st.markdown("---")
        mel = extract_mel(audio, sr)
        left, right = st.columns([3, 2])

        with left:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Waveform</div>", unsafe_allow_html=True)
            fig = plot_waveform(audio, sr)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Mel Spectrogram — model input</div>", unsafe_allow_html=True)
            fig = plot_spectrogram(mel)
            st.pyplot(fig, use_container_width=True); plt.close(fig)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown("<div class='card' style='height:100%'>", unsafe_allow_html=True)
            st.markdown("<div class='section-label'>Detection result</div>", unsafe_allow_html=True)

            if model:
                label, prob = predict(model, mel, threshold)
                color = "#f87171" if label == "FAKE" else "#34d399"
                icon  = "⚠️" if label == "FAKE" else "✅"
                cls   = "verdict-fake" if label == "FAKE" else "verdict-real"

                st.markdown(f"""
                <div style='text-align:center;padding:1.2rem 0 .8rem'>
                  <div style='font-size:2.8rem;margin-bottom:.6rem'>{icon}</div>
                  <div class='{cls}'>{label}</div>
                  <div style='margin-top:1.2rem;font-size:.78rem;color:#4a5a78;
                              text-transform:uppercase;letter-spacing:.08em'>Fake probability</div>
                  <div style='font-family:"IBM Plex Mono",monospace;font-size:2rem;
                              color:{color};font-weight:600;margin-top:.1rem'>
                    {prob*100:.1f}%
                  </div>
                </div>
                <div style='display:flex;justify-content:space-between;
                            font-size:.68rem;color:#4a5a78;margin-bottom:.2rem'>
                  <span>REAL</span><span>FAKE</span>
                </div>
                <div class='bar-track'>
                  <div class='bar-fill' style='width:{int(prob*100)}%;background:{color}'></div>
                </div>
                <div style='font-size:.68rem;color:#3a4a68;text-align:center;margin-bottom:.8rem'>
                  threshold = {threshold:.2f}
                </div>
                """, unsafe_allow_html=True)

                st.markdown("---")
                st.markdown(f"""
                <div>
                  <div class='stat-row'>
                    <span class='stat-label'>Fake probability</span>
                    <span class='stat-val'>{prob:.4f}</span>
                  </div>
                  <div class='stat-row'>
                    <span class='stat-label'>Confidence</span>
                    <span class='stat-val'>{(prob if label=="FAKE" else 1-prob):.4f}</span>
                  </div>
                  <div class='stat-row'>
                    <span class='stat-label'>Input shape</span>
                    <span class='stat-val'>1 × {N_MELS} × {TARGET_FRAMES}</span>
                  </div>
                </div>
                """, unsafe_allow_html=True)

                risk       = "HIGH RISK" if prob >= 0.8 else "MEDIUM RISK" if prob >= 0.5 else "LOW RISK" if prob >= 0.2 else "VERY LOW RISK"
                risk_color = "#f87171" if prob >= 0.8 else "#fbbf24" if prob >= 0.5 else "#34d399"
                st.markdown(f"""
                <div class='risk-box' style='border-color:{risk_color}'>
                  <div class='risk-label'>Risk level</div>
                  <div class='risk-tag' style='color:{risk_color}'>{risk}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='text-align:center;padding:2.5rem 0;color:#3a4a68'>
                  <div style='font-size:2rem;opacity:.5'>🔒</div>
                  <div style='margin-top:.8rem;font-size:.85rem'>Load model weights to run detection</div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with st.expander("🔬 Spectrogram statistics"):
            for col, lbl, v in zip(st.columns(4),
                                   ["Mean", "Std Dev", "Min", "Max"],
                                   [mel.mean(), mel.std(), mel.min(), mel.max()]):
                col.metric(lbl, f"{v:.4f}")


# ── Results ──
elif "Results" in page:
    st.markdown("<h1 style='font-size:1.7rem;margin:0 0 1.4rem'>Model Performance</h1>", unsafe_allow_html=True)

    for col, (val, lbl) in zip(st.columns(5), [
        ("99.89%", "Accuracy"), ("99.93%", "Precision"),
        ("99.85%", "Recall"),   ("99.89%", "F1 Score"), ("100.00%", "ROC-AUC")
    ]):
        col.markdown(f"<div class='pill'><div class='val'>{val}</div><div class='lbl'>{lbl}</div></div>",
                     unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Training curves</div>", unsafe_allow_html=True)
        epochs     = list(range(1, 17))
        train_loss = [.2522,.0682,.0515,.0423,.0358,.0331,.0209,.0186,.0142,.0132,.0191,.0211,.0143,.0119,.0111,.0094]
        val_loss   = [.1870,.1909,.0182,.1098,.0659,.0970,.0062,.0037,.0126,.0043,.0020,.0024,.0133,.0152,.0054,.0070]
        train_acc  = [.8910,.9751,.9825,.9847,.9878,.9900,.9929,.9942,.9952,.9962,.9940,.9934,.9953,.9957,.9968,.9976]
        val_acc    = [.9302,.9466,.9940,.9791,.9862,.9817,.9981,.9993,.9978,.9985,.9985,.9989,.9978,.9978,.9985,.9985]

        fig, axes = plt.subplots(1, 2, figsize=(8, 2.6))
        fig.patch.set_facecolor(BG)
        for ax, y1, y2, ylabel in [
            (axes[0], train_loss, val_loss, "BCE Loss"),
            (axes[1], [a*100 for a in train_acc], [a*100 for a in val_acc], "Accuracy (%)")
        ]:
            ax.plot(epochs, y1, color="#6c8fff", label="Train", linewidth=1.8)
            ax.plot(epochs, y2, color="#a78bfa", label="Val",   linewidth=1.8)
            ax.axvline(11, color="#34d399", linewidth=1, linestyle="--", alpha=0.6, label="Best")
            ax.set_xlabel("Epoch", fontsize=8); ax.set_ylabel(ylabel, fontsize=8)
            ax.legend(fontsize=7, facecolor="#0d0f14", edgecolor=GRID, labelcolor=TCOL)
            style_ax(ax)
        fig.tight_layout(pad=0.6)
        st.pyplot(fig, use_container_width=True); plt.close(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-label'>Confusion matrix — test set (n=2,678)</div>", unsafe_allow_html=True)
        cm = np.array([[1337, 1], [2, 1338]])
        fig, ax = plt.subplots(figsize=(4, 3.0))
        fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
        cmap = mcolors.LinearSegmentedColormap.from_list("cm", ["#0d0f14", "#1a2a5a", "#6c8fff"])
        ax.imshow(cm, cmap=cmap, aspect="auto")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=15,
                        fontweight="bold", fontfamily="IBM Plex Mono",
                        color="#000" if cm[i,j] > cm.max()*0.5 else "#c4d0e8")
        ax.set_xticks([0,1]); ax.set_xticklabels(["Real","Fake"], color=TCOL, fontsize=8)
        ax.set_yticks([0,1]); ax.set_yticklabels(["Real","Fake"], color=TCOL, fontsize=8)
        ax.set_xlabel("Predicted", fontsize=8, color=TCOL)
        ax.set_ylabel("Actual",    fontsize=8, color=TCOL)
        for s in ax.spines.values(): s.set_edgecolor(GRID)
        fig.tight_layout(pad=0.4)
        st.pyplot(fig, use_container_width=True); plt.close(fig)
        st.markdown("""
        <div style='font-size:.78rem;color:#4a5a78;margin-top:.8rem;line-height:1.9'>
          <span style='color:#34d399'>1,337 / 1,338</span> real clips correctly identified<br>
          <span style='color:#34d399'>1,338 / 1,340</span> fake clips correctly identified<br>
          <span style='color:#f87171'>1</span> false positive &nbsp;·&nbsp;
          <span style='color:#f87171'>2</span> false negatives
        </div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<div class='section-label' style='margin-bottom:.8rem'>Architecture — DeepfakeCNN</div>", unsafe_allow_html=True)
    st.markdown("""
    <div class='card'>
    <table style='width:100%;border-collapse:collapse;font-size:.82rem'>
      <thead>
        <tr style='border-bottom:1px solid #252d3d'>
          <th style='text-align:left;padding:.45rem .8rem;color:#4a5a78;font-weight:500'>Layer</th>
          <th style='text-align:left;padding:.45rem .8rem;color:#4a5a78;font-weight:500'>Channels</th>
          <th style='text-align:left;padding:.45rem .8rem;color:#4a5a78;font-weight:500'>Output</th>
          <th style='text-align:left;padding:.45rem .8rem;color:#4a5a78;font-weight:500'>Notes</th>
        </tr>
      </thead>
      <tbody>
        <tr style='border-bottom:1px solid #181c26'><td style='padding:.45rem .8rem;color:#6c8fff;font-family:"IBM Plex Mono",monospace'>Stem</td><td style='padding:.45rem .8rem;color:#8899b8'>1 → 32</td><td style='padding:.45rem .8rem;color:#6677a0'>128 × 87 × 32</td><td style='padding:.45rem .8rem;color:#4a5a78'>Conv3×3 · BN · ReLU</td></tr>
        <tr style='border-bottom:1px solid #181c26'><td style='padding:.45rem .8rem;color:#6c8fff;font-family:"IBM Plex Mono",monospace'>Layer 1</td><td style='padding:.45rem .8rem;color:#8899b8'>32 → 64</td><td style='padding:.45rem .8rem;color:#6677a0'>64 × 44 × 64</td><td style='padding:.45rem .8rem;color:#4a5a78'>2× ResBlock · stride=2</td></tr>
        <tr style='border-bottom:1px solid #181c26'><td style='padding:.45rem .8rem;color:#6c8fff;font-family:"IBM Plex Mono",monospace'>Layer 2</td><td style='padding:.45rem .8rem;color:#8899b8'>64 → 128</td><td style='padding:.45rem .8rem;color:#6677a0'>32 × 22 × 128</td><td style='padding:.45rem .8rem;color:#4a5a78'>2× ResBlock · stride=2</td></tr>
        <tr style='border-bottom:1px solid #181c26'><td style='padding:.45rem .8rem;color:#6c8fff;font-family:"IBM Plex Mono",monospace'>Layer 3</td><td style='padding:.45rem .8rem;color:#8899b8'>128 → 256</td><td style='padding:.45rem .8rem;color:#6677a0'>16 × 11 × 256</td><td style='padding:.45rem .8rem;color:#4a5a78'>2× ResBlock · stride=2</td></tr>
        <tr style='border-bottom:1px solid #181c26'><td style='padding:.45rem .8rem;color:#6c8fff;font-family:"IBM Plex Mono",monospace'>GAP</td><td style='padding:.45rem .8rem;color:#8899b8'>256</td><td style='padding:.45rem .8rem;color:#6677a0'>256</td><td style='padding:.45rem .8rem;color:#4a5a78'>AdaptiveAvgPool2d(1,1)</td></tr>
        <tr><td style='padding:.45rem .8rem;color:#6c8fff;font-family:"IBM Plex Mono",monospace'>Head</td><td style='padding:.45rem .8rem;color:#8899b8'>256 → 1</td><td style='padding:.45rem .8rem;color:#6677a0'>logit</td><td style='padding:.45rem .8rem;color:#4a5a78'>Dropout(0.4) · FC128 · Dropout(0.3) · FC1</td></tr>
      </tbody>
    </table>
    <div style='margin-top:.9rem;font-size:.75rem;color:#3a4a68'>
      2,790,369 parameters &nbsp;·&nbsp; BCEWithLogitsLoss &nbsp;·&nbsp;
      Adam lr=1e-4, wd=1e-5 &nbsp;·&nbsp; ReduceLROnPlateau(factor=0.5, patience=2) &nbsp;·&nbsp; Early stopping patience=5
    </div>
    </div>
    """, unsafe_allow_html=True)


# ── How It Works ──
elif "How It Works" in page:
    st.markdown("<h1 style='font-size:1.7rem;margin:0 0 1.4rem'>How It Works</h1>", unsafe_allow_html=True)

    for num, color, title, desc in [
        ("01", "#6c8fff", "Audio input",          "Any WAV/MP3/FLAC file is loaded as mono and resampled to 16 kHz."),
        ("02", "#a78bfa", "Mel spectrogram",       "Converted to 128 mel bins, log-scaled to dB, then fixed to 87 time frames by padding or cropping."),
        ("03", "#fbbf24", "Z-score normalization", "(x − mean) / std is applied per sample, removing loudness differences so the model focuses on spectral shape."),
        ("04", "#34d399", "Residual CNN",           "7-stage network: stem → 3 pairs of residual blocks (32→64→128→256 channels) → global average pool → classifier head."),
        ("05", "#f87171", "Sigmoid + threshold",   "The output logit passes through sigmoid to get a fake probability in [0, 1]. Classified FAKE if prob ≥ threshold."),
    ]:
        st.markdown(f"""
        <div class='card' style='display:flex;gap:1.2rem;align-items:flex-start'>
          <div class='step-num' style='color:{color};border-color:{color}'>{num}</div>
          <div>
            <div style='font-weight:600;color:#e0e8ff;margin-bottom:.3rem'>{title}</div>
            <div style='font-size:.85rem;color:#8899b8;line-height:1.75'>{desc}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class='card'>
          <div style='font-weight:600;color:#e0e8ff;margin-bottom:.6rem'>Why spectrograms + CNNs?</div>
          <div style='font-size:.84rem;color:#8899b8;line-height:1.8'>
            AI-generated speech leaves artifacts that are invisible in the raw waveform but visible in
            the spectrogram — unnatural formant transitions, absent breath noise, overly regular pitch.
            Treating the spectrogram as an image lets convolutional layers learn these texture patterns
            directly, with no hand-crafted features required.
          </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class='card'>
          <div style='font-weight:600;color:#e0e8ff;margin-bottom:.6rem'>⚠️ Limitations</div>
          <div style='font-size:.84rem;color:#8899b8;line-height:1.8'>
            Trained on the FOR for-2sec dataset (TTS systems from ~2019–2020). Performance may drop
            on newer systems like ElevenLabs or VALL-E, noisy recordings, or phone-channel audio.
            The model detects artifacts from known synthesizers — use it as a screening tool,
            not a definitive verdict.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='card'>
      <div style='font-weight:600;color:#e0e8ff;margin-bottom:.9rem'>Training details</div>
      <div style='display:grid;grid-template-columns:1fr 1fr 1fr;gap:1.2rem'>
        <div>
          <div class='section-label'>Dataset</div>
          <div style='font-size:.82rem;color:#8899b8;line-height:1.9'>
            Fake-or-Real (FOR)<br>for-2sec subset<br>8,935 real + 8,935 fake<br>Split 70 / 15 / 15 %
          </div>
        </div>
        <div>
          <div class='section-label'>Training</div>
          <div style='font-size:.82rem;color:#8899b8;line-height:1.9'>
            30 epochs max<br>Early stop @ epoch 16<br>Batch size 32<br>Adam lr=1e-4
          </div>
        </div>
        <div>
          <div class='section-label'>Augmentation</div>
          <div style='font-size:.82rem;color:#8899b8;line-height:1.9'>
            SpecAugment · p=0.8<br>2× freq mask ≤15 bins<br>2× time mask ≤20 frames<br>Training only
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
