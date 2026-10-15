import { NextResponse } from 'next/server';
import fs from 'fs';
import path from 'path';

export const dynamic = 'force-dynamic';

export async function GET() {
  const filePath = path.join(process.cwd(), 'data', 'live_hardware.json');
  
  if (!fs.existsSync(filePath)) {
    return NextResponse.json({
      connected: false,
      leads_off: true,
      bpm: 0,
      total_beats: 0,
      last_beat: null,
      signal_window: [],
      message: 'ESP32 serial bridge is not running. Launch python -m src.stream_esp32 or connect via Web Serial.'
    });
  }

  try {
    const raw = fs.readFileSync(filePath, 'utf-8');
    const data = JSON.parse(raw);
    return NextResponse.json(data);
  } catch (err) {
    return NextResponse.json(
      { error: 'Failed to read live hardware buffer', details: err.message },
      { status: 500 }
    );
  }
}
