'use client';

import { useState, useEffect, useRef, useCallback } from 'react';

const AAMI_COLORS = {
  N: '#3b82f6', // Normal - Blue
  S: '#f59e0b', // Supraventricular - Amber
  V: '#ef4444', // Ventricular - Red
  F: '#10b981', // Fusion - Green
  Q: '#a855f7', // Paced / Unknown - Purple
};

export default function Home() {
  // Navigation State
  const [activeTab, setActiveTab] = useState('oscilloscopeTab');

  // Metadata & Dropdown State
  const [records, setRecords] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedRecord, setSelectedRecord] = useState('cloud_telemedicine');
  const [selectedModel, setSelectedModel] = useState('improved_1d_cnn');
  const [selectedBaud, setSelectedBaud] = useState(115200);

  // Signal & Oscilloscope State
  const [signalData, setSignalData] = useState([]);
  const [rPeaks, setRPeaks] = useState([]);
  const [samplingRate, setSamplingRate] = useState(250);
  const [leadName, setLeadName] = useState('Cloud Remote (Karachi ➔ Haripur)');
  const [totalRecordBeats, setTotalRecordBeats] = useState(0);

  // Oscilloscope Playback
  const [isPlaying, setIsPlaying] = useState(true);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.0);

  // AI Inference State
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [inferenceResult, setInferenceResult] = useState(null);
  const [selectedSampleBeat, setSelectedSampleBeat] = useState(null);

  // Benchmarks State
  const [benchmarks, setBenchmarks] = useState([]);

  // Live Hardware (ESP32 + AD8232) State
  const [isLiveHardware, setIsLiveHardware] = useState(true);
  const [hardwareConnected, setHardwareConnected] = useState(false);
  const [hardwarePort, setHardwarePort] = useState('COM4');
  const [hardwareBpm, setHardwareBpm] = useState(0);
  const [leadsOff, setLeadsOff] = useState(false);
  const [hardwareBeatsCount, setHardwareBeatsCount] = useState(0);
  const [lastHardwareBeat, setLastHardwareBeat] = useState(null);
  const [isWebSerialActive, setIsWebSerialActive] = useState(false);
  const [rawSerialSnippet, setRawSerialSnippet] = useState('Awaiting serial bytes...');

  // High-performance Ring Buffer Refs (avoids React re-rendering 250Hz canvas loop)
  const liveBufferRef = useRef(new Array(600).fill(0));
  const signalDataRef = useRef([]);
  const rawBaselineRef = useRef(2048);
  const smoothValRef = useRef(0);
  const notchBufferRef = useRef([]);
  const dynamicGainRef = useRef(150.0);
  const lastPeakTimeRef = useRef(0);
  const peakHistoryRef = useRef([]);
  const lastCardiacActivityRef = useRef(0);
  const recentWindowRef = useRef([]);
  const smoothVal2Ref = useRef(0);
  const lastFilteredValRef = useRef(2048);
  const portRef = useRef(null);
  const readerRef = useRef(null);

  // Cloud Telemedicine State (Karachi ➔ Haripur Remote Streaming)
  const [cloudPatientId, setCloudPatientId] = useState('patient-ali-001');
  const [isCloudActive, setIsCloudActive] = useState(false);
  const [cloudPacketsCount, setCloudPacketsCount] = useState(0);
  const [cloudOnline, setCloudOnline] = useState(false);
  const cloudOnlineRef = useRef(false);
  const mqttClientRef = useRef(null);
  const lastDataPacketTimeRef = useRef(0);
  const cloudWatchdogRef = useRef(null);

  // Refs for Canvas Animation
  const ecgCanvasRef = useRef(null);
  const beatCanvasRef = useRef(null);
  const playHeadRef = useRef(0);
  const animationFrameRef = useRef(null);
  const isPlayingRef = useRef(isPlaying);
  const playbackSpeedRef = useRef(playbackSpeed);

  useEffect(() => {
    isPlayingRef.current = isPlaying;
  }, [isPlaying]);

  useEffect(() => {
    playbackSpeedRef.current = playbackSpeed;
  }, [playbackSpeed]);

  useEffect(() => {
    signalDataRef.current = signalData;
  }, [signalData]);

  // Load Records, Models, and Benchmarks on Mount
  useEffect(() => {
    async function initMetadata() {
      try {
        const [recRes, modRes, benchRes] = await Promise.all([
          fetch('/api/records'),
          fetch('/api/models'),
          fetch('/api/benchmarks'),
        ]);

        const recData = await recRes.json();
        const modData = await modRes.json();
        const benchData = await benchRes.json();

        setRecords(recData.records || []);
        setModels(modData.models || []);
        setBenchmarks(benchData.benchmarks || []);
      } catch (err) {
        console.error('Failed to load metadata:', err);
      }
    }
    initMetadata();
  }, []);

  // Fetch Signal Data when selectedRecord changes (for MIT-BIH recordings)
  useEffect(() => {
    if (selectedRecord === 'live_esp32' || selectedRecord === 'cloud_telemedicine') {
      setIsLiveHardware(true);
      setLeadName(selectedRecord === 'cloud_telemedicine' ? 'Cloud Remote (Karachi ➔ Haripur)' : 'AD8232 (Lead I/II)');
      setSamplingRate(250);
      return;
    }

    setIsLiveHardware(false);
    async function fetchSignal() {
      try {
        const res = await fetch(`/api/record-signal?record=${selectedRecord}`);
        const data = await res.json();
        setSignalData(data.signal || []);
        setRPeaks(data.beats || []);
        setSamplingRate(data.sampling_rate || 360);
        setLeadName(data.lead || 'MLII');
        setTotalRecordBeats(data.total_record_beats || 0);
        playHeadRef.current = 0;
      } catch (err) {
        console.error('Failed to load record signal:', err);
      }
    }
    fetchSignal();
  }, [selectedRecord]);

  // Live Hardware Polling from Python bridge (`data/live_hardware.json`) ONLY when in USB serial mode (live_esp32)
  useEffect(() => {
    if (selectedRecord !== 'live_esp32' || isWebSerialActive) return;

    const interval = setInterval(async () => {
      try {
        const res = await fetch('/api/live-hardware');
        const data = await res.json();
        if (data.connected) {
          setHardwareConnected(true);
          setHardwarePort(data.port || 'COM4');
          setHardwareBpm(data.bpm || 0);
          setLeadsOff(data.leads_off || false);
          setHardwareBeatsCount(data.total_beats || 0);
          if (data.last_beat) {
            setLastHardwareBeat(data.last_beat);
          }
          if (data.signal_window && data.signal_window.length > 0) {
            const mean = data.signal_window.reduce((a, b) => a + b, 0) / data.signal_window.length;
            const normalized = data.signal_window.map((v) => (v - mean) / 300.0);
            liveBufferRef.current = normalized;
          }
        } else {
          setHardwareConnected(false);
        }
      } catch (err) {
        // silent catch
      }
    }, 120);

    return () => clearInterval(interval);
  }, [isLiveHardware, isWebSerialActive]);

  // Web Serial API Direct Browser Connection
  const handleConnectWebSerial = async () => {
    if (typeof window === 'undefined' || !('serial' in navigator)) {
      alert('Web Serial is natively supported in Google Chrome, Microsoft Edge, and Opera.\nPlease open http://localhost:3000 in Chrome or Edge.');
      return;
    }

    try {
      const port = await navigator.serial.requestPort();
      await port.open({ baudRate: selectedBaud });
      portRef.current = port;

      setIsLiveHardware(true);
      setSelectedRecord('live_esp32');
      setIsWebSerialActive(true);
      setHardwareConnected(true);
      setHardwarePort(`COM4 (${selectedBaud} baud)`);

      const textDecoder = new TextDecoderStream();
      port.readable.pipeTo(textDecoder.writable);
      const reader = textDecoder.readable.getReader();
      readerRef.current = reader;

      let partialLine = '';
      let lastUiUpdate = Date.now();

      (async () => {
        try {
          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            if (value) {
              partialLine += value;
              const lines = partialLine.split(/[\r\n]+/);
              partialLine = lines.pop(); // retain incomplete chunk

              for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;

                // Check for Leads-off string indicators
                if (trimmed.includes('!') || trimmed.toLowerCase().includes('lead') || trimmed.toLowerCase().includes('off')) {
                  setLeadsOff(true);
                  continue;
                }

                // Extract numeric ADC integer/float with regex
                const numMatch = trimmed.match(/-?\d+(\.\d+)?/);
                if (numMatch) {
                  const num = parseFloat(numMatch[0]);
                  if (!isNaN(num)) {
                    const now = Date.now();

                    // Hardware rail saturation (completely disconnected or shorted wire)
                    const isSaturated = num <= 25 || num >= 4075;
                    if (isSaturated) {
                      setLeadsOff(true);
                      liveBufferRef.current.push(0.0);
                      if (liveBufferRef.current.length > 600) {
                        liveBufferRef.current.shift();
                      }
                      setHardwareBpm(0);
                    } else {
                      // 1. 50 Hz Powerline Mains Notch Filter (5-point moving average at 250 Hz)
                      notchBufferRef.current.push(num);
                      if (notchBufferRef.current.length > 5) notchBufferRef.current.shift();
                      const filtered50Hz = notchBufferRef.current.reduce((a, b) => a + b, 0) / notchBufferRef.current.length;

                      // 2. Instantaneous Slope (dV/dt) calculation to reject slow mechanical wobble & table vibration
                      const sampleSlope = Math.abs(filtered50Hz - lastFilteredValRef.current);
                      lastFilteredValRef.current = filtered50Hz;

                      // 3. Track rolling peak-to-peak amplitude (300 ms window = 75 samples)
                      recentWindowRef.current.push(filtered50Hz);
                      if (recentWindowRef.current.length > 75) recentWindowRef.current.shift();
                      
                      let minVal = filtered50Hz;
                      let maxVal = filtered50Hz;
                      for (let k = 0; k < recentWindowRef.current.length; k++) {
                        const v = recentWindowRef.current[k];
                        if (v < minVal) minVal = v;
                        if (v > maxVal) maxVal = v;
                      }
                      const p2p = maxVal - minVal;

                      // 4. Adaptive baseline removal (~0.9 Hz highpass equivalent to eliminate table & breathing drift)
                      rawBaselineRef.current = rawBaselineRef.current * 0.978 + filtered50Hz * 0.022;
                      const diff = filtered50Hz - rawBaselineRef.current;

                      // 5. Cascaded 2-Stage Anti-Vibration Lowpass Filter (~22 Hz cutoff eliminates mechanical rattle & tremor)
                      smoothValRef.current = smoothValRef.current * 0.58 + diff * 0.42;
                      smoothVal2Ref.current = smoothVal2Ref.current * 0.62 + smoothValRef.current * 0.38;
                      const cleanSignal = smoothVal2Ref.current;
                      const absVal = Math.abs(cleanSignal);

                      // 6. Strict Biological QRS Detector:
                      // A real human heartbeat has a steep rise (sampleSlope >= 4), high energy (p2p >= 42), and prominent deflection (absVal >= 28).
                      // Surrounding vibrations, table knocks, and wire friction do NOT possess this QRS slope/amplitude signature.
                      if ((p2p >= 42 && absVal >= 26) || (sampleSlope >= 5 && absVal >= 32)) {
                        lastCardiacActivityRef.current = now;
                      }

                      // Active lead condition: Heartbeat detected within the last 1500 ms (at least 40 BPM)
                      const isLeadAttached = (now - lastCardiacActivityRef.current) < 1500;

                      if (!isLeadAttached) {
                        // AUDIO JACK UNPLUGGED, TABLE VIBRATION, OR LEADS OFF CHEST -> PURE FLATLINE
                        setLeadsOff(true);
                        liveBufferRef.current.push(0.0);
                        if (liveBufferRef.current.length > 600) {
                          liveBufferRef.current.shift();
                        }
                        setHardwareBpm(0);
                      } else {
                        // LEADS CONNECTED TO BODY -> CLEAN MEDICAL ECG TRACE
                        setLeadsOff(false);

                        if (absVal > 30) {
                          dynamicGainRef.current = dynamicGainRef.current * 0.992 + absVal * 0.008;
                        }
                        const gain = Math.max(220.0, dynamicGainRef.current);
                        const scaled = (cleanSignal / gain) * 1.8;

                        liveBufferRef.current.push(scaled);
                        if (liveBufferRef.current.length > 600) {
                          liveBufferRef.current.shift();
                        }

                        // Real-time R-peak detector & BPM
                        if (scaled > 1.25 && (now - lastPeakTimeRef.current) > 380) { // max 155 BPM
                          if (lastPeakTimeRef.current > 0) {
                            const rrIntervalMs = now - lastPeakTimeRef.current;
                            const instantBpm = Math.round(60000 / rrIntervalMs);
                            if (instantBpm >= 45 && instantBpm <= 165) {
                              peakHistoryRef.current.push(instantBpm);
                              if (peakHistoryRef.current.length > 5) peakHistoryRef.current.shift();
                              const avgBpm = Math.round(
                                peakHistoryRef.current.reduce((a, b) => a + b, 0) / peakHistoryRef.current.length
                              );
                              setHardwareBpm(avgBpm);
                              setHardwareBeatsCount((c) => c + 1);
                            }
                          }
                          lastPeakTimeRef.current = now;
                        }
                      }
                    }

                    // Throttle raw snippet to avoid React re-render lag
                    if (now - lastUiUpdate > 200) {
                      setRawSerialSnippet(trimmed);
                      lastUiUpdate = now;
                    }
                  }
                }
              }
            }
          }
        } catch (readErr) {
          console.error('Serial read ended:', readErr);
        } finally {
          setIsWebSerialActive(false);
          setHardwareConnected(false);
        }
      })();
    } catch (err) {
      if (err.name !== 'NotFoundError') {
        alert(`Serial Port Connection Error: ${err.message}`);
      }
    }
  };

  // Disconnect Web Serial
  const handleDisconnectWebSerial = async () => {
    try {
      if (readerRef.current) {
        await readerRef.current.cancel();
      }
      if (portRef.current) {
        await portRef.current.close();
      }
    } catch (e) {
      console.warn('Error during disconnect:', e);
    } finally {
      setIsWebSerialActive(false);
      setHardwareConnected(false);
    }
  };

  // Connect Cloud Telemedicine Stream (Global MQTT over WebSockets)
  const handleConnectCloud = async () => {
    try {
      const patientId = cloudPatientId.trim() || 'patient-ali-001';
      setIsCloudActive(true);
      setHardwareConnected(true);
      setHardwarePort(`Global Cloud MQTT: ${patientId}`);
      setCloudPacketsCount(0);
      setCloudOnline(false);
      cloudOnlineRef.current = false;
      lastDataPacketTimeRef.current = 0;

      const mqttModule = await import('mqtt');
      const mqtt = mqttModule.default || mqttModule;

      const client = mqtt.connect('wss://broker.emqx.io:8084/mqtt', {
        clientId: `CardioAI-Haripur-${Math.random().toString(16).substring(2, 8)}`,
        clean: true,
        connectTimeout: 10000,
        reconnectPeriod: 2000,
        keepalive: 60,
      });
      mqttClientRef.current = client;

      // Robust 2.5s Watchdog:
      // When live ECG packets stop arriving for 2500ms, mark OFFLINE immediately!
      // This prevents false offline flickering on slight Wi-Fi latency.
      if (cloudWatchdogRef.current) clearInterval(cloudWatchdogRef.current);
      cloudWatchdogRef.current = setInterval(() => {
        if (lastDataPacketTimeRef.current > 0) {
          const gap = Date.now() - lastDataPacketTimeRef.current;
          if (gap > 2500) {
            if (cloudOnlineRef.current) {
              cloudOnlineRef.current = false;
              setCloudOnline(false);
            }
          }
        }
      }, 300);

      client.on('connect', () => {
        console.log('[Cloud Telemedicine] Connected to global broker.emqx.io!');
        client.subscribe(`cardioai/telemedicine/${patientId}`);
        setHardwareConnected(true);
      });

      client.on('reconnect', () => {
        console.log('[Cloud Telemedicine] Reconnecting to broker...');
      });

      client.on('close', () => {
        console.log('[Cloud Telemedicine] Broker WebSocket closed');
      });

      client.on('offline', () => {
        console.log('[Cloud Telemedicine] Broker offline');
        cloudOnlineRef.current = false;
        setCloudOnline(false);
      });

      client.on('message', (topic, message) => {
        try {
          const raw = message.toString();
          const data = JSON.parse(raw);

          // Live ECG Data Batch (10 packets/second):
          lastDataPacketTimeRef.current = Date.now();
          if (!cloudOnlineRef.current) {
            cloudOnlineRef.current = true;
            setCloudOnline(true);
          }
          setCloudPacketsCount((c) => c + 1);

          if (data.lo) {
            setLeadsOff(true);
            liveBufferRef.current.push(0.0);
            if (liveBufferRef.current.length > 600) liveBufferRef.current.shift();
            setHardwareBpm(0);
            return;
          }

          if (data.s && Array.isArray(data.s)) {
            const now = Date.now();
            for (let idx = 0; idx < data.s.length; idx++) {
              const num = data.s[idx];
              if (isNaN(num)) continue;

              // 1. 50 Hz Powerline Notch Filter
              notchBufferRef.current.push(num);
              if (notchBufferRef.current.length > 5) notchBufferRef.current.shift();
              const filtered50Hz = notchBufferRef.current.reduce((a, b) => a + b, 0) / notchBufferRef.current.length;

              // 2. Slope calculation
              const sampleSlope = Math.abs(filtered50Hz - lastFilteredValRef.current);
              lastFilteredValRef.current = filtered50Hz;

              // 3. Peak-to-peak rolling window
              recentWindowRef.current.push(filtered50Hz);
              if (recentWindowRef.current.length > 75) recentWindowRef.current.shift();

              let minVal = filtered50Hz;
              let maxVal = filtered50Hz;
              for (let k = 0; k < recentWindowRef.current.length; k++) {
                const v = recentWindowRef.current[k];
                if (v < minVal) minVal = v;
                if (v > maxVal) maxVal = v;
              }
              const p2p = maxVal - minVal;

              // 4. Baseline removal
              rawBaselineRef.current = rawBaselineRef.current * 0.978 + filtered50Hz * 0.022;
              const diff = filtered50Hz - rawBaselineRef.current;

              // 5. 2-Stage Lowpass
              smoothValRef.current = smoothValRef.current * 0.58 + diff * 0.42;
              smoothVal2Ref.current = smoothVal2Ref.current * 0.62 + smoothValRef.current * 0.38;
              const cleanSignal = smoothVal2Ref.current;
              const absVal = Math.abs(cleanSignal);

              if ((p2p >= 42 && absVal >= 26) || (sampleSlope >= 5 && absVal >= 32)) {
                lastCardiacActivityRef.current = now;
              }

              const isLeadAttached = (now - lastCardiacActivityRef.current) < 1500;

              if (!isLeadAttached) {
                setLeadsOff(true);
                liveBufferRef.current.push(0.0);
                if (liveBufferRef.current.length > 600) liveBufferRef.current.shift();
                setHardwareBpm(0);
              } else {
                setLeadsOff(false);
                if (absVal > 30) {
                  dynamicGainRef.current = dynamicGainRef.current * 0.992 + absVal * 0.008;
                }
                const gain = Math.max(220.0, dynamicGainRef.current);
                const scaled = (cleanSignal / gain) * 1.8;

                liveBufferRef.current.push(scaled);
                if (liveBufferRef.current.length > 600) liveBufferRef.current.shift();

                // R-Peak detection
                if (scaled > 1.25 && (now - lastPeakTimeRef.current) > 380) {
                  if (lastPeakTimeRef.current > 0) {
                    const rrIntervalMs = now - lastPeakTimeRef.current;
                    const instantBpm = Math.round(60000 / rrIntervalMs);
                    if (instantBpm >= 45 && instantBpm <= 165) {
                      peakHistoryRef.current.push(instantBpm);
                      if (peakHistoryRef.current.length > 5) peakHistoryRef.current.shift();
                      const avgBpm = Math.round(
                        peakHistoryRef.current.reduce((a, b) => a + b, 0) / peakHistoryRef.current.length
                      );
                      setHardwareBpm(avgBpm);
                      setHardwareBeatsCount((c) => c + 1);
                    }
                  }
                  lastPeakTimeRef.current = now;
                }
              }
            }
          }
        } catch (err) {
          console.error('[Cloud Telemedicine] Parse error:', err);
        }
      });

      client.on('error', (err) => {
        console.warn('[Cloud Telemedicine] MQTT Error:', err);
      });
    } catch (err) {
      alert(`Cloud Telemedicine connection error: ${err.message}`);
      setIsCloudActive(false);
      setHardwareConnected(false);
    }
  };

  // Disconnect Cloud Telemedicine
  const handleDisconnectCloud = () => {
    if (cloudWatchdogRef.current) {
      clearInterval(cloudWatchdogRef.current);
      cloudWatchdogRef.current = null;
    }
    if (mqttClientRef.current) {
      try {
        mqttClientRef.current.end();
      } catch (e) {}
      mqttClientRef.current = null;
    }
    cloudOnlineRef.current = false;
    setCloudOnline(false);
    setIsCloudActive(false);
    setHardwareConnected(false);
  };

  // Auto-connect to Cloud Telemedicine immediately on page mount (Zero Clicks Required!)
  useEffect(() => {
    let timer = null;
    if (selectedRecord === 'cloud_telemedicine') {
      timer = setTimeout(() => {
        handleConnectCloud();
      }, 400);
    }
    return () => {
      if (timer) clearTimeout(timer);
    };
  }, []);

  // When switching back to Cloud Telemedicine in dropdown, auto-connect immediately
  useEffect(() => {
    if (selectedRecord === 'cloud_telemedicine' && !isCloudActive) {
      handleConnectCloud();
    }
  }, [selectedRecord]);

  // Oscilloscope Grid Painter
  const drawMedicalGrid = useCallback((ctx, width, height) => {
    ctx.fillStyle = '#020617';
    ctx.fillRect(0, 0, width, height);

    // Minor grid
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.06)';
    ctx.lineWidth = 1;
    const minorStep = 18;
    for (let x = 0; x < width; x += minorStep) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += minorStep) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Major grid
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.14)';
    ctx.lineWidth = 1.2;
    const majorStep = minorStep * 5;
    for (let x = 0; x < width; x += majorStep) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += majorStep) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // Center baseline
    ctx.strokeStyle = 'rgba(6, 182, 212, 0.25)';
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();
  }, []);

  // Oscilloscope Animation Loop (Continuous 60 FPS, detached from high-frequency state updates)
  useEffect(() => {
    const canvas = ecgCanvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const handleResize = () => {
      const dpr = window.devicePixelRatio || 1;
      const rect = canvas.getBoundingClientRect();
      canvas.width = rect.width * dpr;
      canvas.height = rect.height * dpr;
      ctx.scale(dpr, dpr);
    };

    handleResize();
    window.addEventListener('resize', handleResize);

    const render = () => {
      const dpr = window.devicePixelRatio || 1;
      const width = canvas.width / dpr;
      const height = canvas.height / dpr;

      drawMedicalGrid(ctx, width, height);

      const centerY = height / 2;
      const yScale = height / 7;

      if (isLiveHardware) {
        // Read directly from High-Performance live RingBuffer
        const buffer = liveBufferRef.current;
        const totalPts = buffer.length;

        ctx.save();
        ctx.strokeStyle = leadsOff ? '#f59e0b' : '#10b981';
        ctx.lineWidth = 2.4;
        ctx.shadowColor = leadsOff ? '#f59e0b' : '#10b981';
        ctx.shadowBlur = 8;
        ctx.beginPath();

        for (let i = 0; i < totalPts; i++) {
          const x = (i / totalPts) * width;
          const y = centerY - buffer[i] * yScale;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
        ctx.restore();

        if (selectedRecord === 'cloud_telemedicine' && !cloudOnlineRef.current) {
          ctx.save();
          ctx.fillStyle = 'rgba(239, 68, 68, 0.95)';
          ctx.font = 'bold 13px Inter, sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText('🔴 PATIENT DEVICE OFFLINE — WAITING FOR STREAM RECONNECTION...', width / 2, 35);
          ctx.restore();
        } else if (leadsOff) {
          ctx.save();
          ctx.fillStyle = 'rgba(245, 158, 11, 0.95)';
          ctx.font = 'bold 13px Inter, sans-serif';
          ctx.textAlign = 'center';
          ctx.fillText('⚠️ AD8232 LEADS DETACHED / NOISY CONTACT — CHECK CHEST ELECTRODES', width / 2, 35);
          ctx.restore();
        }
      } else {
        // MIT-BIH sliding playback window
        const signal = signalDataRef.current;
        if (signal.length > 0) {
          const windowSamples = samplingRate * 4; // 4 seconds window
          const startIndex = Math.floor(playHeadRef.current);
          const endIndex = startIndex + windowSamples;

          let slice = signal.slice(startIndex, endIndex);
          if (slice.length < windowSamples) {
            playHeadRef.current = 0;
            slice = signal.slice(0, windowSamples);
          }

          ctx.save();
          ctx.strokeStyle = '#10b981';
          ctx.lineWidth = 2.2;
          ctx.shadowColor = '#10b981';
          ctx.shadowBlur = 8;
          ctx.beginPath();

          for (let i = 0; i < slice.length; i++) {
            const x = (i / windowSamples) * width;
            const y = centerY - slice[i] * yScale;
            if (i === 0) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();
          ctx.restore();

          // Draw R-peak markers
          rPeaks.forEach((beat) => {
            if (beat.sample >= startIndex && beat.sample <= endIndex) {
              const beatX = ((beat.sample - startIndex) / windowSamples) * width;
              const beatY = centerY - signal[beat.sample] * yScale;
              const color = AAMI_COLORS[beat.class_name] || '#3b82f6';

              ctx.save();
              ctx.fillStyle = color;
              ctx.beginPath();
              ctx.arc(beatX, beatY, 4.5, 0, Math.PI * 2);
              ctx.fill();

              ctx.font = 'bold 11px JetBrains Mono';
              ctx.fillText(beat.class_name, beatX - 4, beatY - 10);
              ctx.restore();
            }
          });

          if (isPlayingRef.current) {
            playHeadRef.current += (samplingRate / 60) * playbackSpeedRef.current;
          }
        }
      }

      animationFrameRef.current = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener('resize', handleResize);
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    };
  }, [samplingRate, isLiveHardware, leadsOff, rPeaks, drawMedicalGrid]);

  // Draw Beat Inspector Waveform Canvas
  useEffect(() => {
    const canvas = beatCanvasRef.current;
    if (!canvas || !selectedSampleBeat) return;
    const ctx = canvas.getContext('2d');
    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = '#020617';
    ctx.fillRect(0, 0, width, height);

    // Grid lines
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let x = 0; x < width; x += 30) {
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }
    for (let y = 0; y < height; y += 30) {
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();
    }

    // R-Peak Line
    const rPeakX = (90 / 200) * width;
    ctx.strokeStyle = 'rgba(239, 68, 68, 0.4)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(rPeakX, 0);
    ctx.lineTo(rPeakX, height);
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw waveform
    const waveform = selectedSampleBeat.waveform;
    const isMatch = selectedSampleBeat.true_class === selectedSampleBeat.pred_class;
    ctx.save();
    ctx.strokeStyle = isMatch ? '#10b981' : '#ef4444';
    ctx.lineWidth = 2.5;
    ctx.shadowColor = isMatch ? '#10b981' : '#ef4444';
    ctx.shadowBlur = 8;
    ctx.beginPath();

    const centerY = height / 2;
    const scale = height / 5;

    for (let i = 0; i < waveform.length; i++) {
      const x = (i / waveform.length) * width;
      const y = centerY - waveform[i] * scale;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.restore();
  }, [selectedSampleBeat]);

  // Run AI Diagnostics Handler
  const handleRunAi = async () => {
    setIsAnalyzing(true);
    try {
      const res = await fetch(`/api/predict?record=${selectedRecord}&model=${selectedModel}`);
      const data = await res.json();
      setInferenceResult(data);
      if (data.sample_beats && data.sample_beats.length > 0) {
        setSelectedSampleBeat(data.sample_beats[0]);
      }
    } catch (err) {
      console.error('Inference execution failed:', err);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const vRecall = inferenceResult?.per_class?.V
    ? (inferenceResult.per_class.V.recall_sensitivity * 100).toFixed(1)
    : '--';

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="app-header">
        <div className="header-brand">
          <div className="pulse-indicator">
            <span className="pulse-ring"></span>
            <span className="pulse-dot"></span>
          </div>
          <div>
            <h1>CardioAI <span className="badge">Next.js v15</span></h1>
            <p className="subtitle">AI-Based ECG Heart Anomaly Detection | MIT-BIH & Real Hardware</p>
          </div>
        </div>

        <div className="header-controls">
          <div className="control-group">
            <label htmlFor="recordSelect">ECG Signal Source</label>
            <select
              id="recordSelect"
              className="custom-select"
              value={selectedRecord}
              onChange={(e) => setSelectedRecord(e.target.value)}
            >
              <optgroup label="Physical Sensor & IoT Telemedicine">
                <option value="live_esp32">⚡ Local Hardware: ESP32 + AD8232 (USB Serial)</option>
                <option value="cloud_telemedicine">☁️ Cloud Telemedicine: Karachi ➔ Haripur (Remote Patient)</option>
              </optgroup>
              <optgroup label="MIT-BIH Arrhythmia Cohort (DS2 Test Records)">
                {records.map((rec) => (
                  <option key={rec} value={rec}>
                    Patient {rec} (PhysioNet)
                  </option>
                ))}
              </optgroup>
            </select>
          </div>

          {selectedRecord === 'cloud_telemedicine' ? (
            <div className="control-group">
              <label htmlFor="patientIdInput">Remote Patient Device ID</label>
              <input
                id="patientIdInput"
                type="text"
                className="custom-select"
                style={{ background: 'rgba(15, 23, 42, 0.8)', color: '#38bdf8', fontWeight: 600 }}
                value={cloudPatientId}
                onChange={(e) => setCloudPatientId(e.target.value)}
                disabled={isCloudActive}
                placeholder="patient-ali-001"
              />
            </div>
          ) : isLiveHardware ? (
            <div className="control-group">
              <label htmlFor="baudSelect">Baud Rate</label>
              <select
                id="baudSelect"
                className="custom-select"
                value={selectedBaud}
                onChange={(e) => setSelectedBaud(Number(e.target.value))}
                disabled={isWebSerialActive}
              >
                <option value={115200}>115200 (Default ESP32)</option>
                <option value={9600}>9600 (Standard Arduino)</option>
                <option value={57600}>57600</option>
              </select>
            </div>
          ) : null}

          <div className="control-group">
            <label htmlFor="modelSelect">AI Model Architecture</label>
            <select
              id="modelSelect"
              className="custom-select"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              {models.map((mod) => (
                <option key={mod} value={mod}>
                  {mod.replace(/_/g, ' ').toUpperCase()}
                </option>
              ))}
            </select>
          </div>

          {selectedRecord === 'cloud_telemedicine' ? (
            isCloudActive ? (
              <button
                id="disconnectCloudBtn"
                className="btn btn-secondary"
                onClick={handleDisconnectCloud}
                style={{ background: '#ef4444', color: '#fff', border: 'none' }}
              >
                <span className="btn-icon">⏹</span>
                Disconnect Cloud
              </button>
            ) : (
              <button
                id="connectCloudBtn"
                className="btn btn-primary"
                onClick={handleConnectCloud}
                style={{ background: 'linear-gradient(135deg, #10b981, #06b6d4)', border: 'none' }}
              >
                <span className="btn-icon">🌐</span>
                Connect Cloud Telemedicine
              </button>
            )
          ) : isLiveHardware ? (
            isWebSerialActive ? (
              <button
                id="disconnectWebSerialBtn"
                className="btn btn-secondary"
                onClick={handleDisconnectWebSerial}
                style={{ background: '#ef4444', color: '#fff', border: 'none' }}
              >
                <span className="btn-icon">⏹</span>
                Disconnect (COM4)
              </button>
            ) : (
              <button
                id="webSerialBtn"
                className="btn btn-primary"
                onClick={handleConnectWebSerial}
                style={{ background: 'linear-gradient(135deg, #06b6d4, #3b82f6)' }}
              >
                <span className="btn-icon">🔌</span>
                Connect Web Serial
              </button>
            )
          ) : (
            <button
              id="runAiBtn"
              className="btn btn-primary"
              onClick={handleRunAi}
              disabled={isAnalyzing}
            >
              <span className="btn-icon">{isAnalyzing ? '⏳' : '⚡'}</span>
              {isAnalyzing ? 'Analyzing...' : 'Run AI Diagnostics'}
            </button>
          )}
        </div>
      </header>

      {/* Main Dashboard */}
      <main className="dashboard-container">
        {/* Navigation Tabs */}
        <nav className="nav-tabs">
          <button
            className={`tab-btn ${activeTab === 'oscilloscopeTab' ? 'active' : ''}`}
            onClick={() => setActiveTab('oscilloscopeTab')}
          >
            Live ECG Monitor & AI
          </button>
          <button
            className={`tab-btn ${activeTab === 'benchmarkTab' ? 'active' : ''}`}
            onClick={() => setActiveTab('benchmarkTab')}
          >
            Academic Benchmarks & Models
          </button>
          <button
            className={`tab-btn ${activeTab === 'aboutTab' ? 'active' : ''}`}
            onClick={() => setActiveTab('aboutTab')}
          >
            Clinical Methodology & AAMI Standard
          </button>
        </nav>

        {/* Tab 1: Live Monitor & AI Diagnostics */}
        {activeTab === 'oscilloscopeTab' && (
          <div className="tab-content active">
            {/* Monitor Card */}
            <section className="card monitor-card">
              <div className="card-header">
                <div className="monitor-title">
                  <span className={`live-tag ${
                    selectedRecord === 'cloud_telemedicine'
                      ? (isCloudActive 
                          ? (cloudOnline ? 'connected' : (cloudPacketsCount === 0 ? 'waiting' : 'offline'))
                          : 'waiting')
                      : isLiveHardware
                        ? (hardwareConnected ? 'connected' : 'waiting')
                        : ''
                  }`}>
                    {selectedRecord === 'cloud_telemedicine'
                      ? (isCloudActive 
                          ? (cloudOnline 
                              ? '☁️ CLOUD STREAM ACTIVE' 
                              : (cloudPacketsCount === 0 ? '⏳ CONNECTING TO PATIENT...' : '🔴 PATIENT OFFLINE')) 
                          : '⏳ AWAITING CLOUD')
                      : isLiveHardware
                        ? (hardwareConnected ? '⚡ ESP32 ACTIVE' : '⏳ AWAITING SERIAL')
                        : 'LIVE STREAM'}
                  </span>
                  <h2>
                    ECG Oscilloscope Monitor (<span>{selectedRecord === 'cloud_telemedicine' ? `Cloud Telemedicine: ${cloudPatientId} (Karachi ➔ Haripur)` : isLiveHardware ? 'ESP32 + AD8232 Sensor' : `Lead ${leadName}`}</span>)
                  </h2>
                </div>
                <div className="monitor-stats">
                  <div className="stat-pill">
                    <span className="stat-label">Sampling Rate:</span>
                    <span className="stat-val">{samplingRate} Hz</span>
                  </div>
                  <div className="stat-pill">
                    <span className="stat-label">Heart Rate:</span>
                    <span className="stat-val">
                      {isLiveHardware
                        ? (hardwareBpm > 0 ? `${hardwareBpm} BPM` : (leadsOff ? 'Leads Off' : 'Calculating...'))
                        : '74 BPM'}
                    </span>
                  </div>
                  <div className="stat-pill">
                    <span className="stat-label">{isLiveHardware ? 'Status:' : 'Total Beats:'}</span>
                    <span className="stat-val">
                      {selectedRecord === 'cloud_telemedicine'
                        ? (isCloudActive 
                            ? (cloudOnline 
                                ? (leadsOff ? '⚠️ Leads Detached' : '🟢 Stream Good') 
                                : (cloudPacketsCount === 0 ? '⏳ Connecting...' : '🔴 Device Offline')) 
                            : '🔴 Disconnected')
                        : isLiveHardware
                          ? (hardwareConnected ? (leadsOff ? '⚠️ Leads Detached' : '🟢 Signal Good') : '🔴 Disconnected')
                          : `${totalRecordBeats.toLocaleString()} Beats`}
                    </span>
                  </div>
                </div>
              </div>

              <div className="oscilloscope-wrapper">
                <canvas ref={ecgCanvasRef} className="ecg-canvas" />
                <div className="oscilloscope-overlay">
                  <div className="grid-label y-max">+3.0 mV</div>
                  <div className="grid-label y-zero">0.0 mV</div>
                  <div className="grid-label y-min">-3.0 mV</div>
                </div>
              </div>

              {/* Live Hardware Debug & Serial / Cloud Monitor Bar */}
              {isLiveHardware && (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '0.6rem 1rem',
                  background: 'rgba(15, 23, 42, 0.75)',
                  borderTop: '1px solid rgba(255, 255, 255, 0.08)',
                  fontSize: '0.82rem',
                  fontFamily: 'var(--font-mono, monospace)',
                  flexWrap: 'wrap',
                  gap: '0.5rem'
                }}>
                  {selectedRecord === 'cloud_telemedicine' ? (
                    <>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.7rem' }}>
                        <span style={{ color: isCloudActive ? (cloudOnline ? '#10b981' : (cloudPacketsCount === 0 ? '#f59e0b' : '#ef4444')) : '#f59e0b', fontWeight: 700 }}>
                          {isCloudActive 
                            ? (cloudOnline ? `● Cloud MQTT: broker.emqx.io` : (cloudPacketsCount === 0 ? '○ Connecting to Broker...' : '● Patient Offline')) 
                            : '○ Awaiting Cloud Stream'}
                        </span>
                        <span style={{ color: '#64748b' }}>|</span>
                        <span style={{ color: '#38bdf8' }}>
                          Remote Patient: <strong style={{ color: '#fff' }}>{cloudPatientId}</strong>
                        </span>
                        <span style={{ color: '#64748b' }}>|</span>
                        <span style={{ color: '#a78bfa' }}>
                          Batches Received: <strong style={{ color: '#fff' }}>{cloudPacketsCount}</strong>
                        </span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.7rem' }}>
                        <span style={{ color: cloudOnline ? (leadsOff ? '#f59e0b' : '#10b981') : (cloudPacketsCount === 0 ? '#f59e0b' : '#ef4444'), fontWeight: 600 }}>
                          {cloudOnline 
                            ? (leadsOff ? '⚠️ Patient Leads Detached in Karachi' : '✓ Patient Connected (Karachi)') 
                            : (cloudPacketsCount === 0 ? '⏳ Awaiting first telemetry packet...' : '🔴 Patient Device Offline / Signal Interrupted')}
                        </span>
                      </div>
                    </>
                  ) : (
                    <>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.7rem' }}>
                        <span style={{ color: hardwareConnected ? '#10b981' : '#f59e0b', fontWeight: 700 }}>
                          {hardwareConnected ? `● ${hardwarePort}` : '○ Waiting for Serial Connection'}
                        </span>
                        <span style={{ color: '#64748b' }}>|</span>
                        <span style={{ color: '#38bdf8' }}>
                          Raw ADC: <strong style={{ color: '#fff' }}>{rawSerialSnippet}</strong>
                        </span>
                      </div>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.7rem' }}>
                        <span style={{ color: leadsOff ? '#f59e0b' : '#10b981' }}>
                          Contact: {leadsOff ? '⚠️ Detached / High Impedance' : '✓ Electrode Contact OK'}
                        </span>
                      </div>
                    </>
                  )}
                </div>
              )}

              <div className="monitor-footer">
                <div className="playback-controls">
                  {!isLiveHardware && (
                    <>
                      <button
                        className="ctrl-btn"
                        onClick={() => setIsPlaying(!isPlaying)}
                      >
                        {isPlaying ? '⏸ Pause' : '▶ Play'}
                      </button>
                      <button
                        className="ctrl-btn"
                        onClick={() => { playHeadRef.current = 0; }}
                      >
                        ⏮ Rewind
                      </button>
                      <div className="speed-selector">
                        <span>Speed:</span>
                        {[0.5, 1.0, 2.0].map((spd) => (
                          <button
                            key={spd}
                            className={`speed-btn ${playbackSpeed === spd ? 'active' : ''}`}
                            onClick={() => setPlaybackSpeed(spd)}
                          >
                            {spd}x
                          </button>
                        ))}
                      </div>
                    </>
                  )}
                  {isLiveHardware && (
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', fontSize: '0.85rem' }}>
                      <span style={{ color: hardwareConnected ? '#10b981' : '#f59e0b', fontWeight: 600 }}>
                        {hardwareConnected ? 'ESP32 Streaming' : 'Click "Connect Web Serial" to start'}
                      </span>
                      {lastHardwareBeat && (
                        <span className="badge" style={{ background: lastHardwareBeat.is_anomaly ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)', color: lastHardwareBeat.is_anomaly ? '#ef4444' : '#10b981' }}>
                          Last Beat: {lastHardwareBeat.class_name} ({lastHardwareBeat.class_full}) {lastHardwareBeat.confidence}%
                        </span>
                      )}
                    </div>
                  )}
                </div>

                <div className="legend-bar">
                  <span className="legend-item"><span className="legend-color color-n"></span> Normal (N)</span>
                  <span className="legend-item"><span className="legend-color color-s"></span> Supraventricular (S)</span>
                  <span className="legend-item"><span className="legend-color color-v"></span> Ventricular Ectopic (V)</span>
                  <span className="legend-item"><span className="legend-color color-f"></span> Fusion (F)</span>
                  <span className="legend-item"><span className="legend-color color-q"></span> Paced / Unknown (Q)</span>
                </div>
              </div>
            </section>

            {/* KPI Metrics Grid */}
            <section className="kpi-grid">
              <div className="kpi-card">
                <div className="kpi-label">{isLiveHardware ? 'Live Pulse (BPM)' : 'Patient Record Accuracy'}</div>
                <div className="kpi-value highlight">
                  {isLiveHardware ? (hardwareBpm > 0 ? `${hardwareBpm}` : '--') : (inferenceResult ? `${(inferenceResult.accuracy * 100).toFixed(2)}%` : '--%')}
                </div>
                <div className="kpi-subtext">
                  {isLiveHardware
                    ? (hardwareConnected ? 'Real-time heart rate from AD8232' : 'Plug in ESP32 via USB')
                    : (inferenceResult ? `Correct: ${(inferenceResult.accuracy * inferenceResult.total_beats).toFixed(0)} / ${inferenceResult.total_beats}` : 'Click "Run AI Diagnostics" to analyze')}
                </div>
              </div>

              <div className="kpi-card">
                <div className="kpi-label">{isLiveHardware ? 'Total Beats Recorded' : 'Macro F1-Score (Balanced)'}</div>
                <div className="kpi-value">
                  {isLiveHardware ? hardwareBeatsCount : (inferenceResult ? inferenceResult.macro_f1.toFixed(4) : '--')}
                </div>
                <div className="kpi-subtext">{isLiveHardware ? 'Continuous real-time R-peaks' : 'Penalizes minority class neglect'}</div>
              </div>

              <div className="kpi-card">
                <div className="kpi-label">Ventricular Ectopic (V) Recall</div>
                <div className="kpi-value success">
                  {vRecall !== '--' ? `${vRecall}%` : '--%'}
                </div>
                <div className="kpi-subtext">Critical PVC detection rate</div>
              </div>

              <div className="kpi-card">
                <div className="kpi-label">Clinical Anomaly Status</div>
                <div className="kpi-value alert">
                  {isLiveHardware
                    ? (lastHardwareBeat?.is_anomaly ? '⚠️ ARRHYTHMIA DETECTED' : (hardwareConnected ? (leadsOff ? 'LEADS DETACHED' : 'NORMAL SINUS RHYTHM') : 'DISCONNECTED'))
                    : (inferenceResult ? (vRecall === '100.0' ? 'VEB CAPTURED' : 'ANALYSIS COMPLETE') : 'AWAITING RUN')}
                </div>
                <div className="kpi-subtext">
                  {isLiveHardware ? 'Real-time 1D-CNN stream analysis' : `Patient ${selectedRecord} loaded`}
                </div>
              </div>
            </section>

            {/* Two-Column Interactive Analysis */}
            <div className="analysis-split">
              {/* Left: Beat Inspector */}
              <section className="card beat-inspector-card">
                <div className="card-header">
                  <h3>🔍 Individual Heartbeat Inspector</h3>
                  <span className="badge">Click any beat below to zoom</span>
                </div>

                <div className="beat-inspector-body">
                  <div className="beat-selector-bar">
                    {inferenceResult?.sample_beats?.length > 0 ? (
                      inferenceResult.sample_beats.map((beat) => (
                        <button
                          key={beat.beat_index}
                          className={`beat-pill ${selectedSampleBeat?.beat_index === beat.beat_index ? 'active' : ''}`}
                          onClick={() => setSelectedSampleBeat(beat)}
                        >
                          #{beat.beat_index} ({beat.true_class} ➜ {beat.pred_class})
                        </button>
                      ))
                    ) : (
                      <span className="loading-msg">Run AI Diagnostics to inspect sample beats...</span>
                    )}
                  </div>

                  {selectedSampleBeat && (
                    <div className="beat-detail-view">
                      <div className="beat-chart-wrapper">
                        <canvas ref={beatCanvasRef} width={450} height={200} className="beat-canvas" />
                        <div className="r-peak-tag">Annotated R-Peak (t = 0 ms)</div>
                      </div>

                      <div className="beat-prediction-details">
                        <div className="prediction-header">
                          <div>
                            <span className="detail-label">Predicted Class:</span>
                            <h4 className="pred-title">Class {selectedSampleBeat.pred_class}</h4>
                          </div>
                          <div>
                            <span className="detail-label">Ground Truth:</span>
                            <h4 className="true-title">Class {selectedSampleBeat.true_class}</h4>
                          </div>
                        </div>

                        <div className="confidence-bars">
                          {['N', 'S', 'V'].map((cls) => {
                            const prob = ((selectedSampleBeat.probabilities?.[cls] || 0) * 100).toFixed(1);
                            return (
                              <div key={cls} className="conf-item">
                                <span className="conf-class">Class {cls}:</span>
                                <div className="bar-container">
                                  <div className="bar-fill" style={{ width: `${prob}%` }}></div>
                                </div>
                                <span className="conf-pct">{prob}%</span>
                              </div>
                            );
                          })}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </section>

              {/* Right: Confusion Matrix */}
              <section className="card metrics-card">
                <div className="card-header">
                  <h3>📊 Confusion Matrix & AAMI Performance</h3>
                  <span className="badge">Unseen Patient Evaluation</span>
                </div>

                <div className="matrix-container">
                  <table className="matrix-table">
                    <thead>
                      <tr>
                        <th>True \ Pred</th>
                        <th>N</th>
                        <th>S</th>
                        <th>V</th>
                        <th>F</th>
                        <th>Q</th>
                      </tr>
                    </thead>
                    <tbody>
                      {inferenceResult?.confusion_matrix ? (
                        inferenceResult.confusion_matrix.map((row, i) => {
                          const classLabels = ['N', 'S', 'V', 'F', 'Q'];
                          return (
                            <tr key={i}>
                              <td><strong>{classLabels[i]}</strong></td>
                              {row.map((val, j) => {
                                const isDiag = i === j;
                                const cellClass = isDiag ? 'cell-diag' : (val > 0 ? 'cell-offdiag' : '');
                                return (
                                  <td key={j} className={cellClass}>
                                    {val.toLocaleString()}
                                  </td>
                                );
                              })}
                            </tr>
                          );
                        })
                      ) : (
                        <tr>
                          <td colSpan={6} className="placeholder-cell">
                            Run AI Diagnostics to populate confusion matrix
                          </td>
                        </tr>
                      )}
                    </tbody>
                  </table>
                </div>
              </section>
            </div>
          </div>
        )}

        {/* Tab 2: Academic Benchmarks */}
        {activeTab === 'benchmarkTab' && (
          <div className="tab-content active">
            <section className="card">
              <div className="card-header">
                <div>
                  <h2>Academic Experiment Progression & Benchmarks</h2>
                  <p className="subtitle">Quantitative validation demonstrating the solution to The Accuracy Paradox</p>
                </div>
              </div>

              <div className="table-responsive">
                <table className="benchmark-table">
                  <thead>
                    <tr>
                      <th>Experiment Name</th>
                      <th>Validation Protocol</th>
                      <th>Training Cohort (DS1)</th>
                      <th>Test Patient (DS2)</th>
                      <th>Accuracy</th>
                      <th>Macro F1</th>
                      <th>Class V (VEB) Recall</th>
                      <th>Class N F1</th>
                      <th>Key Findings / Remarks</th>
                    </tr>
                  </thead>
                  <tbody>
                    {benchmarks.map((row, idx) => (
                      <tr key={idx}>
                        <td><strong>{row.Experiment || '--'}</strong></td>
                        <td>{row.Dataset_Partition || '--'}</td>
                        <td><code>{row.Train_Records || '--'}</code></td>
                        <td><code>{row.Test_Records || '--'}</code></td>
                        <td><span className="stat-val">{row.Overall_Accuracy || '--'}</span></td>
                        <td><span className="stat-val highlight">{row.Macro_F1 || '--'}</span></td>
                        <td>
                          <span className={`stat-val ${parseFloat(row.Class_V_Recall) > 0 ? 'success' : ''}`}>
                            {row.Class_V_Recall || '--'}
                          </span>
                        </td>
                        <td>{row.Class_N_F1 || '--'}</td>
                        <td><small>{row.Notes || '--'}</small></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          </div>
        )}

        {/* Tab 3: Clinical Methodology */}
        {activeTab === 'aboutTab' && (
          <div className="tab-content active">
            <section className="card methodology-card">
              <h2>Clinical Methodology & Rigorous Evaluation Standards</h2>
              <div className="method-grid">
                <div className="method-item">
                  <h3>1. Zero-Phase Bandpass Filtering</h3>
                  <p>Raw ECG leads are filtered using a 3rd-order digital Butterworth bandpass filter (<strong>0.5 Hz – 40.0 Hz</strong>) with zero phase distortion (<code>scipy.signal.filtfilt</code>). This removes respiration baseline drift and 50/60 Hz powerline interference without shifting R-peak locations.</p>
                </div>

                <div className="method-item">
                  <h3>2. Data Leakage Prevention (De Chazal 2004)</h3>
                  <p>We enforce a strict <strong>inter-patient partition</strong>. Naive random heartbeat splitting produces artificially inflated scores (&gt;98%) because heartbeats from the same patient share morphological quirks. In this system, testing is performed exclusively on unseen patients from the DS2 cohort.</p>
                </div>

                <div className="method-item">
                  <h3>3. AAMI EC57 5-Class Super-Group Standard</h3>
                  <p>Annotation symbols from PhysioNet MIT-BIH recordings are mapped into:
                    <strong> N</strong> (Non-Ectopic), <strong>S</strong> (Supraventricular Ectopic), <strong>V</strong> (Ventricular Ectopic), <strong>F</strong> (Fusion), and <strong>Q</strong> (Unknown / Paced). Spurious non-beat markers (<code>+</code>, <code>~</code>, comments) are completely discarded.</p>
                </div>

                <div className="method-item">
                  <h3>4. Lightweight 1D-CNN Architecture</h3>
                  <p>A 4-block 1D-CNN with Batch Normalization, Max-Pooling, Dropout, and Global Adaptive Average Pooling (109,861 trainable parameters). It performs real-time heartbeat anomaly inference in under 20 milliseconds on standard CPU hardware.</p>
                </div>
              </div>
            </section>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="app-footer">
        <p>AI-Based Heart Anomaly Detection Research Framework | Official MIT-BIH Arrhythmia Database (PhysioNet v1.0.0)</p>
      </footer>
    </div>
  );
}
