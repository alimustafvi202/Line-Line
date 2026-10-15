import { NextResponse } from 'next/server';
import ecgData from '@/data/ecgData.json';

export async function GET(request) {
  const { searchParams } = new URL(request.url);
  const recId = searchParams.get('record') || '100';

  const recordObj = ecgData.signals[recId] || ecgData.signals['100'];
  if (!recordObj) {
    return NextResponse.json({ error: `Record ${recId} not found` }, { status: 404 });
  }

  return NextResponse.json(recordObj);
}
