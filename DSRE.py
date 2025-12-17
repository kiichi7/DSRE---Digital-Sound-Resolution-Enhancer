import os
import sys
import traceback
from typing import Optional

import subprocess
import soundfile as sf
import tempfile

import numpy as np
from scipy import signal
import librosa
import resampy

from PySide6 import QtCore, QtWidgets
from PySide6.QtGui import QIcon, QTextCursor

# --- AI 環境偵測 ---
try:
    import torch
    from audiosr import build_model, super_resolution

    HAS_AI = True
except ImportError:
    HAS_AI = False


def add_ffmpeg_to_path():
    if hasattr(sys, "_MEIPASS"):  # 打包後的臨時目錄
        ffmpeg_dir = os.path.join(sys._MEIPASS, "ffmpeg")
    else:
        ffmpeg_dir = os.path.join(os.path.dirname(__file__), "ffmpeg")
    os.environ["PATH"] += os.pathsep + ffmpeg_dir


add_ffmpeg_to_path()


def save_wav24_out(in_path, y_out, sr, out_path, fmt="ALAC", normalize=True):
    # 確保 shape 為 (n, ch)
    if y_out.ndim == 1:
        data = y_out[:, None]
    else:
        data = y_out.T if y_out.shape[0] < y_out.shape[1] else y_out

    data = data.astype(np.float32, copy=False)
    if normalize:
        peak = float(np.max(np.abs(data)))
        if peak > 1.0:
            data /= peak
    else:
        data = np.clip(data, -1.0, 1.0)

    tmp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_wav.close()
    sf.write(tmp_wav.name, data, sr, subtype="FLOAT")

    fmt = fmt.upper()
    out_path = os.path.splitext(out_path)[0] + (".m4a" if fmt == "ALAC" else ".flac")
    codec_map = {"ALAC": "alac", "FLAC": "flac"}
    sample_fmt_map = {"ALAC": "s32p", "FLAC": "s32"}

    if fmt == "ALAC":
        cmd = ["ffmpeg", "-y", "-i", tmp_wav.name, "-i", in_path, "-map", "0:a", "-map", "1:v?", "-map_metadata", "1",
               "-c:a", codec_map[fmt], "-sample_fmt", sample_fmt_map[fmt], "-c:v", "copy", out_path]
    elif fmt == "FLAC":
        cover_tmp = None
        try:
            cover_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
            cover_tmp.close()
            subprocess.run(["ffmpeg", "-y", "-i", in_path, "-an", "-c:v", "copy", cover_tmp.name], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except:
            cover_tmp = None

        if cover_tmp and os.path.exists(cover_tmp.name):
            cmd = ["ffmpeg", "-y", "-i", tmp_wav.name, "-i", in_path, "-i", cover_tmp.name, "-map", "0:a", "-map",
                   "2:v", "-disposition:v", "attached_pic", "-map_metadata", "1", "-c:a", codec_map[fmt], "-sample_fmt",
                   sample_fmt_map[fmt], "-c:v", "copy", out_path]
        else:
            cmd = ["ffmpeg", "-y", "-i", tmp_wav.name, "-i", in_path, "-map", "0:a", "-map_metadata", "1", "-c:a",
                   codec_map[fmt], "-sample_fmt", sample_fmt_map[fmt], out_path]

    subprocess.run(cmd, check=True)
    os.remove(tmp_wav.name)
    if fmt == "FLAC" and cover_tmp and os.path.exists(cover_tmp.name): os.remove(cover_tmp.name)
    return out_path


# ======== AI 處理核心 ========
def ai_process_impl(in_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    if not HAS_AI: raise ImportError("未安裝 AI 依賴組件")
    model = build_model(model_name="audiosr-600k", device=device)
    with torch.no_grad():
        waveform = super_resolution(model, in_path, seed=42, guidance_scale=3.5)
    return waveform.squeeze(), 48000


# ======== DSP 核心 ========
def freq_shift_mono(x: np.ndarray, f_shift: float, d_sr: float) -> np.ndarray:
    N_orig = len(x)
    N_padded = 1 << int(np.ceil(np.log2(max(1, N_orig))))
    S_hilbert = signal.hilbert(np.hstack((x, np.zeros(N_padded - N_orig, dtype=x.dtype))))
    S_factor = np.exp(2j * np.pi * f_shift * d_sr * np.arange(0, N_padded))
    return (S_hilbert * S_factor)[:N_orig].real


def freq_shift_multi(x: np.ndarray, f_shift: float, d_sr: float) -> np.ndarray:
    return np.asarray([freq_shift_mono(x[i], f_shift, d_sr) for i in range(len(x))])


def zansei_impl(x: np.ndarray, sr: int, m: int = 8, decay: float = 1.25, pre_hp: float = 3000.0,
                post_hp: float = 16000.0, filter_order: int = 11, progress_cb=None, abort_cb=None) -> np.ndarray:
    b, a = signal.butter(filter_order, pre_hp / (sr / 2), 'highpass')
    d_src = signal.filtfilt(b, a, x)
    d_sr = 1.0 / sr
    f_dn = freq_shift_mono if (x.ndim == 1) else freq_shift_multi
    d_res = np.zeros_like(x)
    for i in range(m):
        if abort_cb and abort_cb(): break
        shift_hz = sr * (i + 1) / (m * 2.0)
        d_res += f_dn(d_src, shift_hz, d_sr) * np.exp(-(i + 1) * decay)
        if progress_cb: progress_cb(i + 1, m)
    b, a = signal.butter(filter_order, post_hp / (sr / 2), 'highpass')
    d_res = signal.filtfilt(b, a, d_res)
    adj_factor = np.mean(np.abs(x)) / (np.mean(np.abs(d_res)) + np.mean(np.abs(x)) + 1e-12)
    return (x + d_res) * adj_factor


# ======== 後台工作線程 ========
class DSREWorker(QtCore.QThread):
    sig_log = QtCore.Signal(str)
    sig_file_progress = QtCore.Signal(int, int, str)
    sig_step_progress = QtCore.Signal(int, str)
    sig_overall_progress = QtCore.Signal(int, int)
    sig_file_done = QtCore.Signal(str, str)
    sig_error = QtCore.Signal(str, str)
    sig_finished = QtCore.Signal()

    def __init__(self, files, output_dir, params, parent=None):
        super().__init__(parent)
        self.files, self.output_dir, self.params, self._abort = files, output_dir, params, False

    def abort(self):
        self._abort = True

    def run(self):
        total = len(self.files)
        self.sig_overall_progress.emit(0, total)
        for idx, in_path in enumerate(self.files, start=1):
            if self._abort: break
            fname = os.path.basename(in_path)
            self.sig_file_progress.emit(idx, total, fname)
            try:
                if self.params.get("mode") == "AI (Audio-SR)":
                    self.sig_log.emit(f"AI 模式處理中：{fname}")
                    self.sig_step_progress.emit(50, fname)
                    y_out, sr = ai_process_impl(in_path)
                else:
                    y, sr = librosa.load(in_path, mono=False, sr=None)
                    if y.ndim == 1: y = y[np.newaxis, :]
                    target_sr = int(self.params["target_sr"])
                    if sr != target_sr:
                        y = resampy.resample(y, sr, target_sr, filter='kaiser_fast')
                        sr = target_sr
                    y_out = zansei_impl(y, sr, m=int(self.params["m"]), decay=float(self.params["decay"]),
                                        pre_hp=float(self.params["pre_hp"]), post_hp=float(self.params["post_hp"]),
                                        filter_order=int(self.params["filter_order"]),
                                        progress_cb=lambda c, m: self.sig_step_progress.emit(int(c * 100 / m), fname),
                                        abort_cb=lambda: self._abort)

                os.makedirs(self.output_dir, exist_ok=True)
                out_path = os.path.join(self.output_dir, f"ENH_{fname}")
                out_path = save_wav24_out(in_path, y_out, sr, out_path, fmt=self.params['format'])
                self.sig_file_done.emit(in_path, out_path)
            except Exception as e:
                self.sig_error.emit(fname, str(e))
            self.sig_overall_progress.emit(idx, total)
        self.sig_finished.emit()


# ======== GUI ========
class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DSRE v1.2_AI_Beta")
        self.resize(1000, 700)

        # 組件初始化
        self.list_files = QtWidgets.QListWidget()
        self.le_outdir = QtWidgets.QLineEdit(os.path.abspath("output"))
        self.btn_add = QtWidgets.QPushButton("添加文件")
        self.btn_clear = QtWidgets.QPushButton("清空列表")
        self.btn_outdir = QtWidgets.QPushButton("選擇目錄")
        self.btn_start = QtWidgets.QPushButton("開始處理")
        self.btn_cancel = QtWidgets.QPushButton("取消")
        self.btn_cancel.setEnabled(False)

        self.cb_format = QtWidgets.QComboBox()
        self.cb_format.addItems(["ALAC", "FLAC"])
        self.cb_mode = QtWidgets.QComboBox()
        self.cb_mode.addItems(["傳統 DSP (SSB)", "AI (Audio-SR)"])
        if not HAS_AI: self.cb_mode.setItemData(1, 0, QtCore.Qt.UserRole - 1)

        self.sb_m, self.dsb_decay = QtWidgets.QSpinBox(), QtWidgets.QDoubleSpinBox()
        self.sb_m.setRange(1, 128);
        self.sb_m.setValue(8)
        self.dsb_decay.setRange(0.1, 10.0);
        self.dsb_decay.setValue(1.25)
        self.sb_pre, self.sb_post = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        self.sb_pre.setRange(100, 20000);
        self.sb_pre.setValue(3000)
        self.sb_post.setRange(1000, 40000);
        self.sb_post.setValue(16000)
        self.sb_sr, self.sb_order = QtWidgets.QSpinBox(), QtWidgets.QSpinBox()
        self.sb_sr.setRange(44100, 384000);
        self.sb_sr.setValue(96000)
        self.sb_order.setRange(1, 48);
        self.sb_order.setValue(11)

        self.pb_file, self.pb_all = QtWidgets.QProgressBar(), QtWidgets.QProgressBar()
        self.te_log = QtWidgets.QTextEdit();
        self.te_log.setReadOnly(True)
        self.lbl_now = QtWidgets.QLabel("等待中")

        # 佈局
        layout = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QVBoxLayout();
        left.addWidget(QtWidgets.QLabel("輸入清單"));
        left.addWidget(self.list_files)

        mid = QtWidgets.QVBoxLayout();
        mid.addWidget(self.btn_add);
        mid.addWidget(self.btn_clear);
        mid.addSpacing(20)
        mid.addWidget(QtWidgets.QLabel("輸出格式"));
        mid.addWidget(self.cb_format)
        mid.addWidget(QtWidgets.QLabel("演算法模式"));
        mid.addWidget(self.cb_mode);
        mid.addSpacing(20)
        mid.addWidget(QtWidgets.QLabel("輸出路徑"));
        mid.addWidget(self.le_outdir);
        mid.addWidget(self.btn_outdir);
        mid.addStretch()
        mid.addWidget(self.lbl_now);
        mid.addWidget(self.btn_start);
        mid.addWidget(self.btn_cancel)

        right = QtWidgets.QVBoxLayout();
        right.addWidget(QtWidgets.QLabel("參數（僅 DSP 模式）"))
        f = QtWidgets.QFormLayout();
        f.addRow("調製次數", self.sb_m);
        f.addRow("衰減幅度", self.dsb_decay);
        f.addRow("預高通(Hz)", self.sb_pre);
        f.addRow("後高通(Hz)", self.sb_post);
        f.addRow("濾波階數", self.sb_order);
        f.addRow("採樣率", self.sb_sr)
        right.addLayout(f);
        right.addSpacing(20);
        right.addWidget(QtWidgets.QLabel("當前進度"));
        right.addWidget(self.pb_file);
        right.addWidget(QtWidgets.QLabel("總體進度"));
        right.addWidget(self.pb_all);
        right.addWidget(QtWidgets.QLabel("日誌"));
        right.addWidget(self.te_log)

        layout.addLayout(left, 2);
        layout.addLayout(mid, 1);
        layout.addLayout(right, 2)

        # 信號連接
        self.btn_add.clicked.connect(self.on_add);
        self.btn_clear.clicked.connect(self.list_files.clear)
        self.btn_outdir.clicked.connect(lambda: self.le_outdir.setText(QtWidgets.QFileDialog.getExistingDirectory()))
        self.btn_start.clicked.connect(self.on_start);
        self.btn_cancel.clicked.connect(self.on_cancel)

        if not HAS_AI: self.te_log.append("[提示] 未檢測到 AI 環境，已禁用 Audio-SR。")

    def on_add(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "選擇音訊")
        for f in files: self.list_files.addItem(f)

    def on_start(self):
        files = [self.list_files.item(i).text() for i in range(self.list_files.count())]
        if not files: return
        self.btn_start.setEnabled(False);
        self.btn_cancel.setEnabled(True)
        params = {"m": self.sb_m.value(), "decay": self.dsb_decay.value(), "pre_hp": self.sb_pre.value(),
                  "post_hp": self.sb_post.value(), "target_sr": self.sb_sr.value(),
                  "filter_order": self.sb_order.value(), "format": self.cb_format.currentText(),
                  "mode": self.cb_mode.currentText()}
        self.worker = DSREWorker(files, self.le_outdir.text(), params)
        self.worker.sig_log.connect(self.te_log.append);
        self.worker.sig_step_progress.connect(lambda p, f: self.pb_file.setValue(p))
        self.worker.sig_overall_progress.connect(lambda d, t: self.pb_all.setValue(int(d * 100 / t)))
        self.worker.sig_finished.connect(self.on_finished);
        self.worker.start()

    def on_cancel(self):
        if self.worker: self.worker.abort()

    def on_finished(self):
        self.btn_start.setEnabled(True);
        self.btn_cancel.setEnabled(False);
        self.te_log.append("任務結束")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow();
    w.show()
    sys.exit(app.exec())