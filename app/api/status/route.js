import { NextResponse } from 'next/server';

export async function GET() {
  return NextResponse.json({
    status: 'online',
    project: 'ECG-Heart-Anomaly-Research',
    platform: 'Vercel Serverless Edge/Node'
  });
}
