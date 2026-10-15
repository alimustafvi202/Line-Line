import { NextResponse } from 'next/server';
import ecgData from '@/data/ecgData.json';

export async function GET() {
  return NextResponse.json({
    benchmarks: ecgData.benchmarks
  });
}
