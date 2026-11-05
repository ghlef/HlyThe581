/**
 * Converts Float32Array audio sample buffer to 16-bit PCM ArrayBuffer
 */
export function floatTo16BitPCM(input: Float32Array): ArrayBuffer {
  const output = new Int16Array(input.length);
  for (let i = 0; i < input.length; i++) {
    const s = Math.max(-1, Math.min(1, input[i]));
    output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
  }
  return output.buffer;
}

/**
 * Encodes ArrayBuffer to Base64 string
 */
export function arrayBufferToBase64(buffer: ArrayBuffer): string {
  let binary = '';
  const bytes = new Uint8Array(buffer);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return window.btoa(binary);
}

/**
 * Calculates RMS volume (0.0 - 1.0) from Float32Array
 */
export function calculateAudioLevel(samples: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < samples.length; i++) {
    sum += samples[i] * samples[i];
  }
  const rms = Math.sqrt(sum / samples.length);
  return Math.min(1, rms * 5); // Scale for visual feedback
}

/**
 * Format seconds into mm:ss
 */
export function formatTime(totalSeconds: number): string {
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = totalSeconds % 60;
  return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;
}

/**
 * Decodes an uploaded Audio File (e.g. mp3, wav, m4a, ogg, webm) to an AudioBuffer
 */
export async function decodeAudioFile(file: File, sampleRate = 16000): Promise<AudioBuffer> {
  const arrayBuffer = await file.arrayBuffer();
  const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
  const tempCtx = new AudioCtx({ sampleRate });
  try {
    const audioBuffer = await tempCtx.decodeAudioData(arrayBuffer);
    return audioBuffer;
  } finally {
    if (tempCtx.state !== 'closed') {
      tempCtx.close().catch(() => {});
    }
  }
}

/**
 * Converts an AudioBuffer into a 16-bit PCM WAV Blob
 */
export function audioBufferToWavBlob(buffer: AudioBuffer): Blob {
  const numChannels = 1;
  const sampleRate = buffer.sampleRate;
  const format = 1; // PCM
  const bitDepth = 16;
  
  const samples = buffer.getChannelData(0);
  const dataLength = samples.length * (bitDepth / 8);
  const bufferLength = 44 + dataLength;
  const arrayBuffer = new ArrayBuffer(bufferLength);
  const view = new DataView(arrayBuffer);

  function writeString(offset: number, str: string) {
    for (let i = 0; i < str.length; i++) {
      view.setUint8(offset + i, str.charCodeAt(i));
    }
  }

  // RIFF identifier
  writeString(0, 'RIFF');
  // file length minus RIFF header
  view.setUint32(4, 36 + dataLength, true);
  // RIFF type
  writeString(8, 'WAVE');
  // format chunk identifier
  writeString(12, 'fmt ');
  // format chunk length
  view.setUint32(16, 16, true);
  // sample format (raw)
  view.setUint16(20, format, true);
  // channel count
  view.setUint16(22, numChannels, true);
  // sample rate
  view.setUint32(24, sampleRate, true);
  // byte rate (sample rate * block align)
  view.setUint32(28, sampleRate * numChannels * (bitDepth / 8), true);
  // block align (channel count * bytes per sample)
  view.setUint16(32, numChannels * (bitDepth / 8), true);
  // bits per sample
  view.setUint16(34, bitDepth, true);
  // data chunk identifier
  writeString(36, 'data');
  // data chunk length
  view.setUint32(40, dataLength, true);

  // Write PCM samples
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([arrayBuffer], { type: 'audio/wav' });
}

/**
 * Generates an articulated vocal speech simulation AudioBuffer (with speech rhythm, formants, and natural pauses)
 * Used as a fallback when microphone permission is denied or physical mic is not attached.
 */
export function createDemoSpeechBuffer(audioCtx: AudioContext, durationSeconds = 12): AudioBuffer {
  const sampleRate = audioCtx.sampleRate;
  const numSamples = Math.floor(sampleRate * durationSeconds);
  const buffer = audioCtx.createBuffer(1, numSamples, sampleRate);
  const channelData = buffer.getChannelData(0);

  // Speech simulation: syllable bursts with vowel formants (F0=130Hz, F1=650Hz, F2=1700Hz)
  const syllables = [
    { start: 0.5, dur: 0.35, pitch: 140, vowel: 'um' },      // "Um,"
    { start: 1.2, dur: 0.25, pitch: 155, vowel: 'so' },      // "so"
    { start: 1.5, dur: 0.30, pitch: 150, vowel: 'to' },      // "to-"
    { start: 1.8, dur: 0.30, pitch: 142, vowel: 'day' },     // "day"
    { start: 2.3, dur: 0.40, pitch: 135, vowel: 'ba' },      // "ba-"
    { start: 2.7, dur: 0.25, pitch: 148, vowel: 'sic' },     // "sic-"
    { start: 3.0, dur: 0.30, pitch: 140, vowel: 'ly' },      // "ly,"
    { start: 3.7, dur: 0.45, pitch: 130, vowel: 'uh' },      // "uh..."
    { start: 4.5, dur: 0.30, pitch: 160, vowel: 'we' },      // "we"
    { start: 4.9, dur: 0.30, pitch: 150, vowel: 'want' },    // "want"
    { start: 5.3, dur: 0.25, pitch: 145, vowel: 'to' },      // "to"
    { start: 5.8, dur: 0.40, pitch: 158, vowel: 'like' },    // "like"
    { start: 6.5, dur: 0.35, pitch: 145, vowel: 'share' },   // "share"
    { start: 7.0, dur: 0.30, pitch: 140, vowel: 'this' },    // "this"
    { start: 7.4, dur: 0.40, pitch: 155, vowel: 'new' },     // "new"
    { start: 7.9, dur: 0.45, pitch: 138, vowel: 'sys' },     // "sys-"
    { start: 8.4, dur: 0.35, pitch: 130, vowel: 'tem' },     // "tem"
    { start: 9.1, dur: 0.40, pitch: 150, vowel: 'right' },   // "right?"
  ];

  for (const syl of syllables) {
    const startSample = Math.floor(syl.start * sampleRate);
    const endSample = Math.min(numSamples, startSample + Math.floor(syl.dur * sampleRate));
    const sylLength = endSample - startSample;

    for (let i = 0; i < sylLength; i++) {
      const t = i / sampleRate;
      const progress = i / sylLength;
      // Smooth attack and release envelope
      const env = Math.sin(progress * Math.PI);
      
      // Pitch modulation with slight vibrato
      const f0 = syl.pitch * (1 + 0.02 * Math.sin(2 * Math.PI * 5 * t));
      
      // Glottal pulse approximation + vocal tract formants
      const harmonic1 = Math.sin(2 * Math.PI * f0 * t);
      const harmonic2 = 0.5 * Math.sin(2 * Math.PI * f0 * 2 * t);
      const formant1 = 0.4 * Math.sin(2 * Math.PI * 700 * t);
      const formant2 = 0.25 * Math.sin(2 * Math.PI * 1800 * t);
      const aspiration = 0.05 * (Math.random() * 2 - 1);

      const val = (harmonic1 + harmonic2 + formant1 + formant2 + aspiration) * env * 0.35;
      channelData[startSample + i] += val;
    }
  }

  return buffer;
}

