import { createClient } from '@supabase/supabase-js';
import { GoogleGenAI } from '@google/genai';
import { NextResponse } from 'next/server';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_PUBLISHABLE_DEFAULT_KEY!
);
const genAI = new GoogleGenAI({ apiKey: process.env.GOOGLE_API_KEY! });

export async function POST(req: Request) {
  try {
  const { query } = await req.json();

  // Embedding avec Gemini

  const emb = await genAI.models.embedContent({
  model: 'gemini-embedding-001',
  contents: query,
  config: {
    outputDimensionality: 1536,
  },
});
const queryEmbedding = emb.embeddings?.[0]?.values;

    if (!queryEmbedding) {
      return NextResponse.json(
        { error: 'Failed to generate query embedding' },
        { status: 500 }
      );
    }


  // Recherche vectorielle
  const { data: results, error } = await supabase.rpc('match_documents', {
    query_embedding: queryEmbedding,
    match_threshold: 0.0,
    match_count: 5,
  });

  if (error) {
    return NextResponse.json({ error: error.message }, { status: 500 });
  }

  const context = results?.map((r: any) => r.content).join('\n---\n') || '';

  const generationResponse = await genAI.models.generateContent({
      model: 'gemini-3-flash-preview',
      contents: `Context:
${context}

Question:
${query}`,
      config: {
        systemInstruction:
          'You are a helpful assistant. Use the provided context to answer questions. If the answer is not in the context, say you do not know.',
      },
    });

  return NextResponse.json({
    answer: generationResponse.text,
    sources: results,
  });

} catch (error: any) {
  return NextResponse.json({ error: error.message }, { status: 500 });
}
}