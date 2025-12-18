# DSRE / Deep Sound Resolution Enhancer

## 簡介 / Description

DSRE 是一款 **高性能音訊增強工具**，可以將任何音訊檔案批次處理為 **高解析度 (Hi-Res) 音訊**，無需大量計算資源即可快速處理大批次音訊檔案。

DSRE is a **high-performance audio enhancement tool** that can batch-convert any audio files into **high-resolution (Hi-Res) audio**.
Inspired by Sony DSEE HX, it uses a **non-deep-learning frequency enhancement algorithm**, allowing fast processing of large batches without heavy computation.

**主要特性 / Key Features:**

* **批次處理 / Batch Processing**：一次性轉換多個音訊檔案 / Convert multiple audio files at once.
* **多格式支援 / Multiple Formats**：WAV、MP3、FLAC、M4A 等 / Supports WAV, MP3, FLAC, M4A, etc.
* **保留封面與元資料 / Preserves Cover & Metadata**：無需手動修改 / No manual editing required.
* **彈性參數控制 / Flexible Parameters**：調變次數、衰減幅度、高通濾波器等 / Modulation count, decay, high-pass filter, etc.
* **快速穩定 / Fast & Stable**：不依賴深度學習模型 / Does not rely on deep learning, fast processing.
* **多語言支援 / Multi-language Support**：支援繁體中文與英文介面 / Supports Traditional Chinese and English interfaces.

---

## 安裝與使用 / Installation & Usage

---

## 參數說明 / Parameters

| 參數 / Parameter                               | 預設值 / Default | 說明 / Description                                                   |
| -------------------------------------------- | ------------- | ------------------------------------------------------------------ |
| 語言 / Language                              | 繁體中文 / English | 選擇介面語言 / Choose the interface language.                        |
| 調變次數 (m) / Modulation count                  | 8             | 音訊增強重複次數 / Number of enhancement repetitions, higher = more detail |
| 衰減幅度 (decay)                                 | 1.25          | 高頻衰減控制 / High-frequency decay control                              |
| 前置處理高通截止頻率 / Pre-processing high-pass cutoff  | 3000 Hz       | 處理前高通濾波器 / Pre-enhancement high-pass filter                        |
| 後置處理高通截止頻率 / Post-processing high-pass cutoff | 16000 Hz      | 處理後高通濾波器 / Post-enhancement high-pass filter                       |
| 濾波器階數 / Filter order                         | 11            | 高通濾波器階數 / High-pass filter order                                   |
| 目標取樣率 / Target sampling rate                 | 96000 Hz      | 輸出音訊取樣率 / Output audio sample rate                                 |
| 輸出格式 / Output format                         | ALAC / FLAC   | 選擇 Hi-Res 輸出格式 / Choose Hi-Res output format                       |

---
