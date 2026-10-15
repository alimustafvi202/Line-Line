import { NextResponse } from 'next/server';
import ecgData from '@/data/ecgData.json';

export async function POST(request) {
  try {
    const body = await request.json();
    const recId = body.record || '100';
    const model = body.model || 'improved_1d_cnn';

    const key = `${recId}_${model}`;
    const predictionObj = ecgData.predictions[key] || ecgData.predictions[`100_${model}`];

    if (!predictionObj) {
      return NextResponse.json(
        { error: `Inference combination ${key} not found` },
        { status: 404 }
      );
    }

    return NextResponse.json(predictionObj);
  } catch (err) {
    return NextResponse.json(
      { error: 'Invalid prediction request body' },
      { status: 400 }
    );
  }
}
