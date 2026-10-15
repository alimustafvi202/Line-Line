"""
ESP32 + AD8232 Real-Time ECG Serial Bridge and AI Classifier
Author: Antigravity Agent
Project: CardioAI ECG Heart Anomaly Detection

This script connects to the ESP32 via Serial (CP2102 USB Bridge), reads the raw analog
ADC ECG data from the AD8232 sensor, applies the clinical 0.5-40 Hz bandpass filter,
performs real-time R-peak detection to calculate live BPM, and classifies heartbeats
using the trained PyTorch 1D-CNN model (AAMI EC57 classes: N, S, V, F, Q).

The live buffer is written to `data/live_hardware.json` so the Next.js web application
can stream it directly onto the oscilloscope monitor.
"""

import sys
import os
import time
import json
import argparse
import numpy as np
import scipy.signal as signal
import serial
import serial.tools.list_ports

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
from src.model import ECGCNN1D

AAMI_CLASSES = ['N', 'S', 'V', 'F', 'Q']
AAMI_NAMES = {
    'N': 'Normal Sinus Beat',
    'S': 'Supraventricular Ectopic',
    'V': 'Ventricular Ectopic (PVC)',
    'F': 'Fusion of Ventricular & Normal',
    'Q': 'Paced / Unclassifiable'
}

def auto_detect_com_port():
    """Scan and return the most likely COM port for ESP32 / CP2102."""
    ports = list(serial.tools.list_ports.comports())
    if not ports:
        return None
    
    # Priority keywords for ESP32 CP2102 USB bridge
    keywords = ['cp210', 'silicon labs', 'uart', 'usb serial', 'ch340', 'esp32']
    for p in ports:
        desc = f"{p.device} {p.description} {p.manufacturer}".lower()
        if any(kw in desc for kw in keywords):
            return p.device
            
    # Default to first available COM port
    return ports[0].device

def apply_bandpass_filter(data, lowcut=0.5, highcut=40.0, fs=250.0, order=3):
    """Apply zero-phase Butterworth bandpass filter."""
    if len(data) < 30:
        return np.array(data, dtype=np.float32)
    nyq = 0.5 * fs
    low = max(0.01, lowcut / nyq)
    high = min(0.99, highcut / nyq)
    b, a = signal.butter(order, [low, high], btype='band')
    try:
        # Use lfilter or filtfilt if enough samples
        if len(data) > 3 * max(len(a), len(b)):
            return signal.filtfilt(b, a, data)
        return signal.lfilter(b, a, data)
    except Exception:
        return np.array(data, dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description="ESP32 AD8232 Live ECG AI Inference Bridge")
    parser.add_argument("--port", type=str, default=None, help="COM port (e.g. COM3). Auto-detected if omitted.")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate (default: 115200)")
    parser.add_argument("--model", type=str, default="outputs/models/improved_1d_cnn.pt", help="PyTorch model weights path")
    args = parser.parse_args()

    # Find port
    port = args.port or auto_detect_com_port()
    if not port:
        print("\n[ERROR] No active COM port detected!")
        print("Please ensure the CP2102 driver is installed and the ESP32 is firmly connected via USB.")
        print("Run 'Install_CP2102_Driver.bat' as Administrator to install the driver.\n")
        return

    print("=" * 70)
    print("      CARDIO-AI: ESP32 + AD8232 REAL-TIME ECG STREAMING BRIDGE")
    print("=" * 70)
    print(f"Target Serial Port: {port} @ {args.baud} baud")

    # Load PyTorch Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = os.path.join(PROJECT_ROOT, args.model)
    model = ECGCNN1D(num_classes=5)
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print(f"Loaded AI Model: {args.model}")
    else:
        print(f"[WARNING] Model weights not found at {model_path}. Using initial weights.")
    model.to(device)
    model.eval()

    # Prepare JSON output directory
    os.makedirs(os.path.join(PROJECT_ROOT, "data"), exist_ok=True)
    live_json_path = os.path.join(PROJECT_ROOT, "data", "live_hardware.json")

    # Open Serial Port
    try:
        ser = serial.Serial(port, args.baud, timeout=1.0)
        time.sleep(1.5) # Allow ESP32 boot/reset
        ser.reset_input_buffer()
        print(f"[SUCCESS] Connected to {port}! Streaming live ECG signals...\n")
    except Exception as e:
        print(f"[CONNECTION ERROR] Could not open {port}: {e}")
        return

    raw_buffer = []
    filtered_buffer = []
    max_buf_len = 1500 # ~6 seconds buffer at 250 Hz
    last_r_peak_sample = 0
    beat_count = 0
    recent_rr_intervals = []
    current_bpm = 0.0
    last_beat_info = None
    leads_off = False
    last_write_time = time.time()

    try:
        while True:
            line = ser.readline().decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            # Check for Leads-Off indicator (! or text from AD8232 LO+/LO- pins)
            if '!' in line or 'lead' in line.lower() or 'off' in line.lower():
                leads_off = True
                val = 0.0
            else:
                try:
                    val = float(line)
                    # AD8232 ADC range: typically 0 - 4095 (12-bit ESP32 ADC)
                    if val <= 10 or val >= 4085:
                        leads_off = True
                    else:
                        leads_off = False
                except ValueError:
                    continue

            raw_buffer.append(val)
            if len(raw_buffer) > max_buf_len:
                raw_buffer.pop(0)

            # Filter signal periodically or on buffer update
            if len(raw_buffer) >= 50:
                filt = apply_bandpass_filter(raw_buffer, fs=250.0)
                filtered_buffer = filt.tolist()

                # Detect R-peaks if not leads-off
                if not leads_off and len(filtered_buffer) > 200:
                    recent_slice = np.array(filtered_buffer[-250:])
                    mean_val = np.mean(recent_slice)
                    std_val = np.std(recent_slice) + 1e-6
                    norm_slice = (recent_slice - mean_val) / std_val

                    # Peak finding: minimum distance ~100 samples (0.4s at 250Hz -> max 150 BPM)
                    peaks, properties = signal.find_peaks(norm_slice, height=1.2, distance=80)
                    
                    if len(peaks) > 0:
                        latest_peak_idx = len(filtered_buffer) - 250 + peaks[-1]
                        if latest_peak_idx - last_r_peak_sample > 80:
                            # New R-peak detected!
                            if last_r_peak_sample > 0:
                                rr_samples = latest_peak_idx - last_r_peak_sample
                                rr_sec = rr_samples / 250.0
                                if 0.3 <= rr_sec <= 2.0:
                                    bpm = 60.0 / rr_sec
                                    recent_rr_intervals.append(bpm)
                                    if len(recent_rr_intervals) > 5:
                                        recent_rr_intervals.pop(0)
                                    current_bpm = round(float(np.median(recent_rr_intervals)), 1)

                            last_r_peak_sample = latest_peak_idx
                            beat_count += 1

                            # Segment 200 samples around R-peak (90 before, 110 after)
                            start_idx = latest_peak_idx - 90
                            end_idx = latest_peak_idx + 110
                            if start_idx >= 0 and end_idx <= len(filtered_buffer):
                                segment = np.array(filtered_buffer[start_idx:end_idx], dtype=np.float32)
                                seg_mean = np.mean(segment)
                                seg_std = np.std(segment) + 1e-6
                                norm_seg = (segment - seg_mean) / seg_std

                                # Model inference
                                tensor_in = torch.tensor(norm_seg).unsqueeze(0).unsqueeze(0).to(device)
                                with torch.no_grad():
                                    logits = model(tensor_in)
                                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                                    pred_class_idx = int(np.argmax(probs))
                                    pred_class = AAMI_CLASSES[pred_class_idx]
                                    conf = float(probs[pred_class_idx])

                                last_beat_info = {
                                    "beat_number": beat_count,
                                    "class_name": pred_class,
                                    "class_full": AAMI_NAMES[pred_class],
                                    "confidence": round(conf * 100, 1),
                                    "waveform": [round(float(x), 3) for x in norm_seg.tolist()],
                                    "is_anomaly": pred_class in ['V', 'S', 'F'],
                                    "timestamp": round(time.time(), 3)
                                }

                                status_indicator = "🟢 [NORMAL]" if pred_class == 'N' else "🔴 [ARRHYTHMIA ALERT]"
                                print(f"{status_indicator} Beat #{beat_count} | Class: {pred_class} ({AAMI_NAMES[pred_class]}) | Conf: {round(conf*100,1)}% | BPM: {current_bpm}")

            # Write to JSON buffer every ~50ms
            now = time.time()
            if now - last_write_time >= 0.05:
                last_write_time = now
                payload = {
                    "connected": True,
                    "port": port,
                    "baud": args.baud,
                    "leads_off": leads_off,
                    "bpm": current_bpm,
                    "total_beats": beat_count,
                    "last_beat": last_beat_info,
                    "signal_window": [round(float(x), 2) for x in (filtered_buffer[-500:] if filtered_buffer else raw_buffer[-500:])],
                    "timestamp": round(now, 3)
                }
                temp_path = live_json_path + ".tmp"
                with open(temp_path, "w") as f:
                    json.dump(payload, f)
                os.replace(temp_path, live_json_path)

    except KeyboardInterrupt:
        print("\nStopping streaming bridge...")
    finally:
        ser.close()
        # Mark as disconnected
        payload = {"connected": False, "port": port}
        with open(live_json_path, "w") as f:
            json.dump(payload, f)
        print("Serial port closed cleanly.")

if __name__ == "__main__":
    main()
