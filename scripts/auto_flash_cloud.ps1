$cli = "d:\Website\ECG-Heart-Anomaly-Research\tools\arduino-cli\arduino-cli.exe"
$sketch = "d:\Website\ECG-Heart-Anomaly-Research\firmware\esp32_cloud_telemedicine"
$fqbn = "esp32:esp32:esp32:UploadSpeed=115200,FlashMode=dio,FlashFreq=40"
$port = "COM4"

Write-Host "=========================================================="
Write-Host "  CARDIO-AI: CLOUD TELEMEDICINE ESP32 FIRMWARE FLASHER    "
Write-Host "=========================================================="
Write-Host "1. Compiling Cloud Telemedicine ECG firmware..."
& $cli compile --fqbn $fqbn $sketch

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n[ERROR] Compilation failed."
    exit 1
}

Write-Host "`n2. Uploading Cloud Telemedicine firmware to $port..."
& $cli upload -p $port --fqbn $fqbn $sketch

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n[SUCCESS] Cloud Telemedicine Firmware flashed successfully!"
    Write-Host "ESP32 will now connect to Wi-Fi and stream to Global Cloud MQTT!"
} else {
    Write-Host "`n[NOTICE] If upload timed out, hold down the 'BOOT' button on ESP32 and try again."
}
