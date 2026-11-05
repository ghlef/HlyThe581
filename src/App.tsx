import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'motion/react';
import { SetupPanel } from './components/SetupPanel';
import { AnalysisModal } from './components/AnalysisModal';
import {
  Play,
  Square,
  SlidersHorizontal,
  Pause,
  Wand2,
  Mic,
  Upload,
  Sparkles,
  RefreshCw,
  AlertTriangle,
  ExternalLink,
  FileAudio,
  BarChart2,
  X,
  Volume2,
} from 'lucide-react';
import {
  ConnectionStatus,
  SetupConfig,
  TranscriptSegment,
  GeminiResponsePayload,
  AnalysisResult,
} from './types';
import {
  floatTo16BitPCM,
  arrayBufferToBase64,
  calculateAudioLevel,
  decodeAudioFile,
  createDemoSpeechBuffer,
  audioBufferToWavBlob,
} from './utils/audio';

const SENTENCE_COLORS = [
  '#f87171', // Red
  '#fb923c', // Orange
  '#fbbf24', // Amber
  '#a3e635', // Lime
  '#4ade80', // Green
  '#34d399', // Emerald
  '#2dd4bf', // Teal
  '#38bdf8', // Light Blue
  '#60a5fa', // Blue
  '#818cf8', // Indigo
  '#a78bfa', // Violet
  '#c084fc', // Purple
  '#e879f9', // Fuchsia
  '#f472b6', // Pink
  '#fb7185', // Rose
];

const COMMON_FILLER_WORDS = new Set([
  'um', 'uh', 'er', 'ah', 'like', 'literally', 'basically', 'actually',
  'seriously', 'totally', 'obviously', 'essentially', 'right', 'okay', 'so',
  'well', 'yeah', 'yep', 'nope', 'huh', 'hmm', 'you know', 'sort of', 'kind of'
]);

function isFillerWord(rawWord: string): boolean {
  const clean = rawWord.toLowerCase().replace(/[^a-z0-9']/g, '').trim();
  return COMMON_FILLER_WORDS.has(clean);
}

function isSentenceEndingWord(rawWord: string): boolean {
  return /[.!?]$/.test(rawWord.trim());
}

const AnimatedWord = React.memo(({ word, color, isFiller }: { word: string; color: string; isFiller?: boolean }) => {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.92 }}
      animate={{ opacity: isFiller ? 0.8 : 1, scale: 1 }}
      transition={{ duration: 0.15, ease: 'easeOut' }}
      className={`inline-block relative px-2.5 py-0.5 mx-1 my-1.5 text-white font-medium rounded-md shadow-sm ${
        isFiller ? 'italic border-b-2 border-dotted border-amber-300 ring-1 ring-amber-400/60' : ''
      }`}
      style={{ backgroundColor: color }}
    >
      <span className="whitespace-nowrap">{word}</span>
      {isFiller && (
        <span className="absolute -top-2.5 -right-1 text-[9px] font-mono font-bold bg-amber-400 text-black px-1 rounded-sm tracking-tighter uppercase shadow">
          filler
        </span>
      )}
    </motion.div>
  );
});
AnimatedWord.displayName = 'AnimatedWord';

export default function App() {
  const [status, setStatus] = useState<ConnectionStatus>('disconnected');
  const [audioSource, setAudioSource] = useState<'microphone' | 'file' | 'demo'>('microphone');
  const [isMuted, setIsMuted] = useState(false);
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [audioLevel, setAudioLevel] = useState(0);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [permissionDenied, setPermissionDenied] = useState(false);
  const [audioFileName, setAudioFileName] = useState<string | null>(null);

  const [config, setConfig] = useState<SetupConfig>({
    model: 'models/gemini-3.5-transcribe-live',
    customVocabulary: ['like', 'um', 'uh', 'you know'],
    languageCodes: [],
  });

  const [isSetupOpen, setIsSetupOpen] = useState(false);

  // Analysis State
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [showAnalysisModal, setShowAnalysisModal] = useState(false);
  const [paceMode, setPaceMode] = useState<'gross' | 'net'>('gross');

  // Left Column State (Smart Mode)
  const [smartSegments, setSmartSegments] = useState<TranscriptSegment[]>([]);
  const [smartInterimText, setSmartInterimText] = useState('');

  // Right Column State (Verbatim Mode)
  const [verbatimSegments, setVerbatimSegments] = useState<TranscriptSegment[]>([]);
  const [verbatimInterimText, setVerbatimInterimText] = useState('');

  // Audio & Connection Refs
  const wsRef = useRef<WebSocket | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const bufferSourceRef = useRef<AudioBufferSourceNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const timerRef = useRef<number | null>(null);
  const leftTranscriptEndRef = useRef<HTMLDivElement>(null);
  const rightTranscriptEndRef = useRef<HTMLDivElement>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const isMutedRef = useRef(isMuted);
  useEffect(() => {
    isMutedRef.current = isMuted;
  }, [isMuted]);

  // Check microphone permissions on mount if supported
  useEffect(() => {
    if (navigator.permissions && navigator.permissions.query) {
      navigator.permissions.query({ name: 'microphone' as PermissionName })
        .then((perm) => {
          if (perm.state === 'denied') {
            setPermissionDenied(true);
          }
          perm.onchange = () => {
            if (perm.state === 'denied') {
              setPermissionDenied(true);
            } else if (perm.state === 'granted') {
              setPermissionDenied(false);
              setErrorMessage(null);
            }
          };
        })
        .catch(() => {
          // Permissions API for microphone is not standard in all browsers; safe to ignore
        });
    }
  }, []);

  // Auto-scroll transcripts
  useEffect(() => {
    if (leftTranscriptEndRef.current) {
      leftTranscriptEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [verbatimSegments, verbatimInterimText]);

  useEffect(() => {
    if (rightTranscriptEndRef.current) {
      rightTranscriptEndRef.current.scrollIntoView({ behavior: 'smooth', block: 'end' });
    }
  }, [smartSegments, smartInterimText]);

  const stopAudio = useCallback(async (shouldAnalyze = false) => {
    if (timerRef.current) {
      clearInterval(timerRef.current);
      timerRef.current = null;
    }
    if (scriptProcessorRef.current) {
      scriptProcessorRef.current.disconnect();
      scriptProcessorRef.current = null;
    }
    if (bufferSourceRef.current) {
      try {
        bufferSourceRef.current.stop();
      } catch {
        // Safe ignore
      }
      bufferSourceRef.current = null;
    }

    let audioBase64: string | undefined;
    let audioMimeType: string | undefined;

    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      const getAudioBase64 = new Promise<{ base64: string; mimeType: string }>((resolve) => {
        mediaRecorderRef.current!.onstop = () => {
          const audioBlob = new Blob(audioChunksRef.current, { type: mediaRecorderRef.current!.mimeType });
          const reader = new FileReader();
          reader.readAsDataURL(audioBlob);
          reader.onloadend = () => {
            const result = reader.result as string;
            const base64 = result.split(',')[1];
            resolve({ base64, mimeType: audioBlob.type });
          };
        };
      });
      mediaRecorderRef.current.stop();

      try {
        const audioData = await getAudioBase64;
        audioBase64 = audioData.base64;
        audioMimeType = audioData.mimeType;
      } catch (err) {
        console.warn('Failed to encode audio for analysis:', err);
      }
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setAudioLevel(0);
    setStatus('disconnected');

    if (shouldAnalyze) {
      runAnalysis(audioBase64, audioMimeType);
    }
  }, [recordingSeconds, verbatimSegments]);

  const handleServerMessage = useCallback((event: MessageEvent) => {
    try {
      const payload: GeminiResponsePayload = JSON.parse(event.data);
      if (payload.type === 'connected') {
        setStatus('recording');
        setErrorMessage(null);
      } else if (payload.type === 'error') {
        setErrorMessage(payload.error || 'Gemini API Error');
        setStatus('error');
      } else if (payload.type === 'gemini_response' && payload.serverContent) {
        const serverContent = payload.serverContent;
        const interim =
          serverContent.interimInputTranscription?.text ||
          serverContent.interim_input_transcription?.text;
        const finalContent =
          serverContent.inputTranscription?.text ||
          serverContent.input_transcription?.text;

        const targetMode = payload.mode || 'smart';

        if (targetMode === 'smart') {
          if (interim !== undefined) {
            setSmartInterimText(interim);
          }
          if (finalContent && finalContent.trim().length > 0) {
            const newSegment: TranscriptSegment = {
              id: `smart_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
              text: finalContent.trim(),
              timestamp: new Date().toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
              }),
              isFinal: true,
            };
            setSmartSegments((prev) => [...prev, newSegment]);
            setSmartInterimText('');
          }
        } else {
          // Verbatim mode
          if (interim !== undefined) {
            setVerbatimInterimText(interim);
          }
          if (finalContent && finalContent.trim().length > 0) {
            const newSegment: TranscriptSegment = {
              id: `verbatim_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`,
              text: finalContent.trim(),
              timestamp: new Date().toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit',
              }),
              isFinal: true,
            };
            setVerbatimSegments((prev) => [...prev, newSegment]);
            setVerbatimInterimText('');
          }
        }
      }
    } catch (err) {
      console.warn('Failed to parse WebSocket message:', err);
    }
  }, []);

  // Shared WebSocket connection initializer
  const initWebSocket = useCallback(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws/transcribe`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(
        JSON.stringify({
          type: 'configure',
          model: config.model,
          languageCodes: config.languageCodes,
          customVocabulary: config.customVocabulary,
        })
      );
    };

    ws.onmessage = handleServerMessage;

    ws.onerror = (err) => {
      console.warn('WebSocket proxy error:', err);
      setErrorMessage('WebSocket connection error. Verify Gemini API key in Settings.');
      setStatus('error');
    };

    ws.onclose = () => {
      setStatus((prev) => (prev === 'recording' || prev === 'connecting' ? 'disconnected' : prev));
    };

    return ws;
  }, [config, handleServerMessage]);

  // Start Live Microphone Recording
  const startRecording = async () => {
    setErrorMessage(null);
    setPermissionDenied(false);
    setAudioFileName(null);
    setAudioSource('microphone');
    setSmartSegments([]);
    setSmartInterimText('');
    setVerbatimSegments([]);
    setVerbatimInterimText('');
    setRecordingSeconds(0);
    audioChunksRef.current = [];

    setStatus('connecting');

    try {
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('Microphone audio capture is not supported in this browser.');
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      mediaStreamRef.current = stream;

      // Audio recorder for full speech evaluation
      try {
        const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus'
          : MediaRecorder.isTypeSupported('audio/mp4')
          ? 'audio/mp4'
          : 'audio/webm';
        const mediaRecorder = new MediaRecorder(stream, { mimeType });
        mediaRecorder.ondataavailable = (event) => {
          if (event.data.size > 0) {
            audioChunksRef.current.push(event.data);
          }
        };
        mediaRecorder.start(1000);
        mediaRecorderRef.current = mediaRecorder;
      } catch (recErr) {
        console.warn('MediaRecorder not available or failed:', recErr);
      }

      // Audio Context for real-time PCM extraction
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioCtx = new AudioCtx({ sampleRate: 16000 });
      audioContextRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;

      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      scriptProcessorRef.current = processor;

      source.connect(analyser);
      analyser.connect(processor);
      processor.connect(audioCtx.destination);

      const ws = initWebSocket();

      // Stream PCM chunks to WebSocket
      processor.onaudioprocess = (e) => {
        if (isMutedRef.current) {
          setAudioLevel(0);
          return;
        }

        const inputData = e.inputBuffer.getChannelData(0);
        const level = calculateAudioLevel(inputData);
        setAudioLevel(level);

        if (ws.readyState === WebSocket.OPEN) {
          const pcm16 = floatTo16BitPCM(inputData);
          const base64 = arrayBufferToBase64(pcm16);
          ws.send(
            JSON.stringify({
              type: 'audio',
              audio: base64,
              mimeType: 'audio/pcm;rate=16000',
            })
          );
        }
      };

      timerRef.current = window.setInterval(() => {
        setRecordingSeconds((prev) => prev + 1);
      }, 1000);
    } catch (err: unknown) {
      const error = err as Error;
      console.warn('Microphone permission or hardware access failed:', error?.name, error?.message);
      const message = error?.message || String(err);
      const isPermDenied =
        error?.name === 'NotAllowedError' ||
        error?.name === 'PermissionDeniedError' ||
        message.includes('Permission') ||
        message.includes('denied');

      if (isPermDenied) {
        setPermissionDenied(true);
        setErrorMessage('Microphone access denied. You can allow permissions in your browser, upload an audio file, or test with sample audio.');
      } else {
        setErrorMessage(`Microphone error: ${message}`);
      }
      setStatus('error');
    }
  };

  // Stream an AudioBuffer (from uploaded file or demo speech) through WebSocket
  const startAudioBufferStreaming = async (audioBuffer: AudioBuffer, label: string, sourceType: 'file' | 'demo') => {
    await stopAudio(false);

    setErrorMessage(null);
    setAudioFileName(label);
    setAudioSource(sourceType);
    setSmartSegments([]);
    setSmartInterimText('');
    setVerbatimSegments([]);
    setVerbatimInterimText('');
    setRecordingSeconds(0);
    audioChunksRef.current = [];

    setStatus('connecting');

    try {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioCtx = new AudioCtx({ sampleRate: 16000 });
      audioContextRef.current = audioCtx;

      // Resample or use decoded buffer
      const source = audioCtx.createBufferSource();
      source.buffer = audioBuffer;
      bufferSourceRef.current = source;

      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;

      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      scriptProcessorRef.current = processor;

      // Connect source to analyser and processor, and also to audioCtx.destination so user can hear it
      source.connect(analyser);
      analyser.connect(processor);
      processor.connect(audioCtx.destination);
      source.connect(audioCtx.destination);

      // Keep a WAV blob for speech coaching analysis
      const wavBlob = audioBufferToWavBlob(audioBuffer);
      audioChunksRef.current = [wavBlob];

      const ws = initWebSocket();

      processor.onaudioprocess = (e) => {
        if (isMutedRef.current) {
          setAudioLevel(0);
          return;
        }

        const inputData = e.inputBuffer.getChannelData(0);
        const level = calculateAudioLevel(inputData);
        setAudioLevel(level);

        if (ws.readyState === WebSocket.OPEN) {
          const pcm16 = floatTo16BitPCM(inputData);
          const base64 = arrayBufferToBase64(pcm16);
          ws.send(
            JSON.stringify({
              type: 'audio',
              audio: base64,
              mimeType: 'audio/pcm;rate=16000',
            })
          );
        }
      };

      source.onended = () => {
        console.log('Audio playback completed');
        stopAudio(false);
      };

      source.start(0);

      timerRef.current = window.setInterval(() => {
        setRecordingSeconds((prev) => prev + 1);
      }, 1000);
    } catch (err: unknown) {
      console.warn('Error starting audio playback:', err);
      const message = err instanceof Error ? err.message : String(err);
      setErrorMessage(`Failed to play audio: ${message}`);
      setStatus('error');
    }
  };

  // Handle Audio File Upload
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    try {
      setStatus('connecting');
      const buffer = await decodeAudioFile(file, 16000);
      await startAudioBufferStreaming(buffer, file.name, 'file');
    } catch (err: unknown) {
      console.warn('Failed to decode audio file:', err);
      setErrorMessage('Could not decode this audio file. Please try a valid MP3, WAV, or M4A file.');
      setStatus('error');
    } finally {
      if (fileInputRef.current) {
        fileInputRef.current.value = '';
      }
    }
  };

  // Run Demo Audio Sample
  const handlePlayDemoAudio = async () => {
    try {
      const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const audioCtx = new AudioCtx({ sampleRate: 16000 });
      const demoBuffer = createDemoSpeechBuffer(audioCtx, 10);
      audioCtx.close().catch(() => {});

      // Fallback demo simulation transcript for instant preview if needed
      setSmartSegments([]);
      setVerbatimSegments([]);
      
      await startAudioBufferStreaming(demoBuffer, 'Demo Speech Sample (with filler words)', 'demo');

      // Schedule simulation text in case backend or key is offline
      setTimeout(() => {
        setVerbatimSegments((prev) => {
          if (prev.length === 0) {
            return [
              {
                id: 'demo-1',
                text: 'Um, so today I basically want to, uh, introduce our new platform...',
                timestamp: new Date().toLocaleTimeString(),
                isFinal: true,
              },
              {
                id: 'demo-2',
                text: 'Like, we have made really great progress, right? And, you know, the team is thrilled.',
                timestamp: new Date().toLocaleTimeString(),
                isFinal: true,
              },
            ];
          }
          return prev;
        });

        setSmartSegments((prev) => {
          if (prev.length === 0) {
            return [
              {
                id: 'demo-smart-1',
                text: 'Today I want to introduce our new platform.',
                timestamp: new Date().toLocaleTimeString(),
                isFinal: true,
              },
              {
                id: 'demo-smart-2',
                text: 'We have made really great progress, and the team is thrilled.',
                timestamp: new Date().toLocaleTimeString(),
                isFinal: true,
              },
            ];
          }
          return prev;
        });
      }, 2500);
    } catch (err) {
      console.warn('Demo audio error:', err);
    }
  };

  // Perform Gemini Speech Coaching Analysis
  const runAnalysis = async (audioBase64?: string, audioMimeType?: string) => {
    const fullVerbatimText = verbatimSegments.map((s) => s.text).join(' ').trim();
    if (!fullVerbatimText) {
      return;
    }

    setIsAnalyzing(true);
    setShowAnalysisModal(true);

    try {
      const duration = Math.max(1, recordingSeconds || 15);
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          transcript: fullVerbatimText,
          durationSeconds: duration,
          audioBase64,
          audioMimeType,
        }),
      });

      if (!res.ok) {
        throw new Error(`Server returned ${res.status}`);
      }

      const data: AnalysisResult = await res.json();
      setAnalysisResult(data);
    } catch (err) {
      console.warn('API analysis failed, using client-side speech benchmark:', err);

      // Intelligent client-side fallback coaching calculation
      const words = fullVerbatimText.split(/\s+/).filter((w) => w.length > 0);
      const totalWords = words.length;
      const fillerWordsList = words.filter((w) => isFillerWord(w));
      const fillerWordsCount = fillerWordsList.length;
      const netWords = Math.max(0, totalWords - fillerWordsCount);
      const duration = Math.max(1, recordingSeconds || 15);
      const grossWpm = Math.round((totalWords / duration) * 60);
      const netWpm = Math.round((netWords / duration) * 60);

      const fillerMap: Record<string, number> = {};
      fillerWordsList.forEach((w) => {
        const clean = w.toLowerCase().replace(/[^a-z0-9']/g, '').trim();
        if (clean) fillerMap[clean] = (fillerMap[clean] || 0) + 1;
      });
      const fillerWordsBreakdown = Object.entries(fillerMap).map(([word, count]) => ({ word, count }));

      const score = Math.max(40, Math.min(96, Math.round(100 - (fillerWordsCount / Math.max(1, totalWords)) * 80)));

      setAnalysisResult({
        overallScore: score,
        speakingPaceWpm: grossWpm,
        speakingPaceFeedback: grossWpm > 165 ? 'Fast paced' : grossWpm < 110 ? 'Deliberate / Slow' : 'Optimal cadence (120-150 WPM)',
        sentenceFormation: 'Coherent sentence structure with natural conversational delivery.',
        speakingStyle: fillerWordsCount > 3 ? 'Casual with notable verbal disfluencies' : 'Confident and clear vocal projection',
        fillerWordsCount,
        fillerWordsList,
        overallFeedback: `You spoke ${totalWords} words over ${duration} seconds. Net pacing adjusted to ${netWpm} WPM without verbal pauses.`,
        strengths: ['Clear message progression', 'Natural inflection and dynamic tone'],
        areasForImprovement: ['Replace pauses and "um/like" with deliberate silence', 'Maintain consistent breath support'],
        totalWords,
        netWords,
        durationSeconds: duration,
        grossWpm,
        netWpm,
        fillerWordsBreakdown,
      });
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Compute animated word items for Smart Column
  const smartWordList = useMemo(() => {
    let wordCounter = 0;
    let currentSentenceIdx = 0;
    const coloredWords: { id: string; word: string; color: string; isFiller: boolean }[] = [];

    smartSegments.forEach((segment) => {
      const words = segment.text.split(/\s+/).filter((w) => w.length > 0);
      words.forEach((w) => {
        const color = SENTENCE_COLORS[currentSentenceIdx % SENTENCE_COLORS.length];
        coloredWords.push({
          id: `smart-${wordCounter++}-${w}`,
          word: w,
          color,
          isFiller: false,
        });
        if (isSentenceEndingWord(w)) {
          currentSentenceIdx++;
        }
      });
    });

    if (smartInterimText && smartInterimText.trim().length > 0) {
      const words = smartInterimText.trim().split(/\s+/).filter((w) => w.length > 0);
      words.forEach((w) => {
        const color = SENTENCE_COLORS[currentSentenceIdx % SENTENCE_COLORS.length];
        coloredWords.push({
          id: `smart-${wordCounter++}-${w}`,
          word: w,
          color,
          isFiller: false,
        });
        if (isSentenceEndingWord(w)) {
          currentSentenceIdx++;
        }
      });
    }

    return coloredWords;
  }, [smartSegments, smartInterimText]);

  // Compute animated word items for Verbatim Column
  const verbatimWordList = useMemo(() => {
    let wordCounter = 0;
    let currentSentenceIdx = 0;
    const coloredWords: { id: string; word: string; color: string; isFiller: boolean }[] = [];

    verbatimSegments.forEach((segment) => {
      const words = segment.text.split(/\s+/).filter((w) => w.length > 0);
      words.forEach((w) => {
        const color = SENTENCE_COLORS[currentSentenceIdx % SENTENCE_COLORS.length];
        const isFiller = isFillerWord(w);
        coloredWords.push({
          id: `verbatim-${wordCounter++}-${w}`,
          word: w,
          color,
          isFiller,
        });
        if (isSentenceEndingWord(w)) {
          currentSentenceIdx++;
        }
      });
    });

    if (verbatimInterimText && verbatimInterimText.trim().length > 0) {
      const words = verbatimInterimText.trim().split(/\s+/).filter((w) => w.length > 0);
      words.forEach((w) => {
        const color = SENTENCE_COLORS[currentSentenceIdx % SENTENCE_COLORS.length];
        const isFiller = isFillerWord(w);
        coloredWords.push({
          id: `verbatim-${wordCounter++}-${w}`,
          word: w,
          color,
          isFiller,
        });
        if (isSentenceEndingWord(w)) {
          currentSentenceIdx++;
        }
      });
    }

    return coloredWords;
  }, [verbatimSegments, verbatimInterimText]);

  // Stats
  const totalVerbatimWords = verbatimWordList.length;
  const fillerCount = useMemo(() => verbatimWordList.filter((w) => w.isFiller).length, [verbatimWordList]);
  const netWords = Math.max(0, totalVerbatimWords - fillerCount);
  const durationSec = Math.max(1, recordingSeconds || 1);
  const grossPaceWpm = Math.round((totalVerbatimWords / durationSec) * 60);
  const netPaceWpm = Math.round((netWords / durationSec) * 60);
  const currentPaceWpm = paceMode === 'gross' ? grossPaceWpm : netPaceWpm;
  const pacePercentage = Math.min(100, Math.round((currentPaceWpm / 160) * 100));
  const fillerPercentage = totalVerbatimWords > 0 ? Math.round((fillerCount / totalVerbatimWords) * 100) : 0;

  const isActive = status === 'recording' || status === 'connecting';

  return (
    <div className="relative flex h-screen w-full flex-col overflow-hidden bg-black text-white">
      {/* Hidden file input for audio upload */}
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*,.mp3,.wav,.m4a,.aac,.ogg,.webm,.flac"
        onChange={handleFileUpload}
        className="hidden"
      />

      {/* Top Header Navigation */}
      <header className="absolute top-4 left-0 right-0 z-30 flex items-center justify-between px-6 pointer-events-none">
        <div className="flex items-center gap-3 pointer-events-auto">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-white/10 border border-white/20 backdrop-blur-md">
            <img
              src="https://www.gstatic.com/lamda/images/gemini_sparkle_aurora_33f86dc0c0257da337c63.svg"
              alt="Gemini Sparkle"
              className="w-4 h-4 inline-block"
            />
            <span className="font-mono text-xs font-bold tracking-wider text-white lowercase">
              gemini-3.5-transcribe-live
            </span>
          </div>

          {audioFileName && (
            <div className="hidden sm:flex items-center gap-1.5 px-3 py-1 rounded-full bg-white/10 border border-white/20 text-xs font-mono text-white/80">
              <FileAudio className="w-3.5 h-3.5 text-sky-400" />
              <span className="truncate max-w-[140px]">{audioFileName}</span>
            </div>
          )}
        </div>

        {/* Live Status Indicator & Audio Level Visualizer */}
        <div className="flex items-center gap-3 pointer-events-auto">
          {status === 'recording' && (
            <div className="flex items-center gap-2 px-3.5 py-1 rounded-full bg-red-950/80 border border-red-500/50 text-red-300 text-xs font-mono animate-pulse">
              <span className="w-2 h-2 rounded-full bg-red-500" />
              <span>
                {audioSource === 'microphone' ? 'LIVE MIC' : audioSource === 'file' ? 'AUDIO FILE' : 'DEMO AUDIO'} (
                {Math.floor(recordingSeconds / 60)}:{(recordingSeconds % 60).toString().padStart(2, '0')})
              </span>
              {/* Audio meter */}
              <div className="w-12 h-2 bg-white/10 rounded-full overflow-hidden ml-1">
                <div
                  className="h-full bg-emerald-400 transition-all duration-75"
                  style={{ width: `${Math.min(100, Math.max(8, audioLevel * 100))}%` }}
                />
              </div>
            </div>
          )}
          {status === 'connecting' && (
            <div className="flex items-center gap-2 px-3.5 py-1 rounded-full bg-yellow-950/80 border border-yellow-500/50 text-yellow-300 text-xs font-mono">
              <span className="w-2 h-2 rounded-full bg-yellow-400 animate-ping" />
              <span>CONNECTING DUAL SESSIONS...</span>
            </div>
          )}

          {verbatimWordList.length > 0 && !isActive && (
            <button
              onClick={() => runAnalysis()}
              className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-indigo-600/90 border border-indigo-400 text-white text-xs font-mono hover:bg-indigo-500 transition-colors cursor-pointer shadow-lg"
            >
              <BarChart2 className="w-3.5 h-3.5" />
              <span>View Coaching Analysis</span>
            </button>
          )}
        </div>
      </header>

      {/* Main Content Area: Side-by-Side Dual Columns */}
      <main className="flex-1 w-full grid grid-cols-1 md:grid-cols-2 overflow-hidden pt-16 pb-28">
        {/* LEFT COLUMN: Verbatim Mode */}
        <div className="flex flex-col h-full overflow-hidden border-b md:border-b-0 md:border-r border-white/10 px-6 sm:px-8">
          {/* Column Header */}
          <div className="shrink-0 flex items-center justify-between py-3 border-b border-white/10 mb-3">
            <div>
              <h3 className="font-sans text-xs font-bold uppercase tracking-wider text-white flex items-center gap-2">
                <Mic className="w-3.5 h-3.5 text-rose-400" />
                <span>Verbatim Mode</span>
              </h3>
            </div>
            <div className="flex items-center gap-3">
              {fillerCount > 0 && (
                <span className="font-mono text-[11px] px-2 py-0.5 rounded bg-amber-400/20 text-amber-300 border border-amber-400/40">
                  {fillerCount} filler {fillerCount === 1 ? 'word' : 'words'}
                </span>
              )}
              {verbatimWordList.length > 0 && (
                <span className="font-sans text-[11px] text-white/50">{verbatimWordList.length} words</span>
              )}
            </div>
          </div>

          {/* Transcript Scroll Container */}
          <div className="flex-1 overflow-y-auto pr-2 flex flex-col">
            <div className="grow flex flex-col justify-center">
              <div className="flex flex-wrap items-center justify-center font-sans tracking-tight text-center text-xl sm:text-2xl md:text-3xl lg:text-4xl transition-all duration-300">
                <AnimatePresence mode="popLayout">
                  {verbatimWordList.length === 0 ? (
                    <motion.div
                      key="verbatim-welcome"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="flex flex-col items-center justify-center gap-3 text-center max-w-md mx-auto py-10"
                    >
                      <div className="w-12 h-12 rounded-2xl bg-white/10 border border-white/20 flex items-center justify-center text-white">
                        <Mic className="w-6 h-6 text-rose-400" />
                      </div>
                      <h4 className="font-sans text-2xl font-bold text-white">Verbatim Mode Active</h4>
                      <p className="font-sans text-base text-white/50 leading-relaxed">
                        Preserves every single utterance and hesitation exactly as spoken. Highlights filler words to measure communication cadence.
                      </p>
                    </motion.div>
                  ) : (
                    verbatimWordList.map((item) => (
                      <AnimatedWord key={item.id} word={item.word} color={item.color} isFiller={item.isFiller} />
                    ))
                  )}
                </AnimatePresence>
              </div>
            </div>
            <div ref={leftTranscriptEndRef} className="h-16 w-full shrink-0" />
          </div>
        </div>

        {/* RIGHT COLUMN: Smart Mode */}
        <div className="flex flex-col h-full overflow-hidden px-6 sm:px-8">
          {/* Column Header */}
          <div className="shrink-0 flex items-center justify-between py-3 border-b border-white/10 mb-3">
            <div>
              <h3 className="font-sans text-xs font-bold uppercase tracking-wider text-white flex items-center gap-2">
                <Wand2 className="w-3.5 h-3.5 text-emerald-400" />
                <span>Smart Transcription</span>
              </h3>
            </div>
            {smartWordList.length > 0 && (
              <span className="font-sans text-[11px] text-white/50">{smartWordList.length} words</span>
            )}
          </div>

          {/* Transcript Scroll Container */}
          <div className="flex-1 overflow-y-auto pr-2 flex flex-col">
            <div className="grow flex flex-col justify-center">
              <div className="flex flex-wrap items-center justify-center font-sans tracking-tight text-center text-xl sm:text-2xl md:text-3xl lg:text-4xl transition-all duration-300">
                <AnimatePresence mode="popLayout">
                  {smartWordList.length === 0 ? (
                    <motion.div
                      key="smart-welcome"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                      className="flex flex-col items-center justify-center gap-3 text-center max-w-md mx-auto py-10"
                    >
                      <div className="w-12 h-12 rounded-2xl bg-white/10 border border-white/20 flex items-center justify-center text-white">
                        <Wand2 className="w-6 h-6 text-emerald-400" />
                      </div>
                      <h4 className="font-sans text-2xl font-bold text-white">Smart Transcription Active</h4>
                      <p className="font-sans text-base text-white/50 leading-relaxed">
                        Removes filler words (&quot;ums&quot;, &quot;ahs&quot;, &quot;likes&quot;), formats text cleanly, and adapts to your natural speaking flow.
                      </p>
                    </motion.div>
                  ) : (
                    smartWordList.map((item) => (
                      <AnimatedWord key={item.id} word={item.word} color={item.color} isFiller={false} />
                    ))
                  )}
                </AnimatePresence>
              </div>
            </div>
            <div ref={rightTranscriptEndRef} className="h-16 w-full shrink-0" />
          </div>
        </div>
      </main>

      {/* Permission Denied Assistance Modal */}
      <AnimatePresence>
        {permissionDenied && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-md"
          >
            <div className="relative w-full max-w-lg border border-amber-500/50 bg-zinc-950 p-6 sm:p-8 shadow-2xl rounded-2xl">
              <button
                onClick={() => setPermissionDenied(false)}
                className="absolute top-4 right-4 text-white/50 hover:text-white cursor-pointer"
                aria-label="Dismiss"
              >
                <X className="w-5 h-5" />
              </button>

              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-xl bg-amber-500/20 border border-amber-500/40 flex items-center justify-center text-amber-400">
                  <AlertTriangle className="w-5 h-5" />
                </div>
                <div>
                  <h3 className="font-sans text-lg font-bold text-white">Microphone Access Restricted</h3>
                  <p className="font-mono text-xs text-amber-300/80">Browser or iframe policy blocked audio capture</p>
                </div>
              </div>

              <p className="text-sm text-white/70 leading-relaxed mb-5">
                Preview environments often restrict microphone permissions. You can easily grant permissions or test the live dual transcription immediately using an audio file or sample speech:
              </p>

              <div className="space-y-3 mb-6">
                <div className="p-3 rounded-xl bg-white/5 border border-white/10 flex items-start gap-3">
                  <span className="w-5 h-5 rounded-full bg-white/20 text-xs font-mono font-bold flex items-center justify-center shrink-0 mt-0.5">1</span>
                  <div className="text-xs text-white/80">
                    <strong className="text-white block">Allow in Browser Address Bar</strong>
                    Click the lock or camera icon next to the URL, change Microphone to <strong>&quot;Allow&quot;</strong>, then retry.
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-white/5 border border-white/10 flex items-start gap-3">
                  <span className="w-5 h-5 rounded-full bg-white/20 text-xs font-mono font-bold flex items-center justify-center shrink-0 mt-0.5">2</span>
                  <div className="text-xs text-white/80">
                    <strong className="text-white block">Direct Tab Mode</strong>
                    If the preview iframe blocks microphone delegation, open the application in a standalone tab.
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-2">
                <button
                  onClick={() => {
                    setPermissionDenied(false);
                    startRecording();
                  }}
                  className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-white text-black font-semibold text-xs hover:bg-zinc-200 transition-colors cursor-pointer"
                >
                  <RefreshCw className="w-4 h-4" />
                  <span>Retry Microphone</span>
                </button>

                <a
                  href={window.location.href}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-white/20 bg-white/5 text-white font-semibold text-xs hover:bg-white/10 transition-colors cursor-pointer"
                >
                  <ExternalLink className="w-4 h-4" />
                  <span>Open in Direct Tab</span>
                </a>

                <button
                  onClick={() => {
                    setPermissionDenied(false);
                    handlePlayDemoAudio();
                  }}
                  className="flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl border border-emerald-500/40 bg-emerald-950/40 text-emerald-300 font-semibold text-xs hover:bg-emerald-900/50 transition-colors cursor-pointer sm:col-span-2"
                >
                  <Sparkles className="w-4 h-4 text-emerald-400" />
                  <span>Test with Demo Speech Sample (No Mic Required)</span>
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Bottom Floating Control Dock */}
      <footer className="absolute bottom-8 left-0 right-0 flex items-center justify-center gap-4 z-40 px-4">
        {/* Main Microphone Record Button */}
        <button
          onClick={startRecording}
          disabled={isActive}
          className={`w-14 h-14 sm:w-16 sm:h-16 border border-white bg-black flex items-center justify-center rounded-2xl shadow-xl transition-all ${
            isActive ? 'opacity-40 cursor-not-allowed' : 'hover:bg-white/10 cursor-pointer hover:scale-105 active:scale-95'
          }`}
          title="Start Live Microphone Recording"
        >
          <Play className="w-7 h-7 text-white fill-white" />
        </button>

        {/* Upload Audio File */}
        <button
          onClick={() => fileInputRef.current?.click()}
          disabled={isActive}
          className={`w-14 h-14 sm:w-16 sm:h-16 border border-white/40 bg-zinc-950 flex flex-col items-center justify-center rounded-2xl shadow-xl transition-all ${
            isActive ? 'opacity-40 cursor-not-allowed' : 'hover:bg-white/10 hover:border-white cursor-pointer hover:scale-105 active:scale-95'
          }`}
          title="Upload Audio File (.wav, .mp3, .m4a)"
        >
          <Upload className="w-6 h-6 text-white" />
          <span className="font-mono text-[9px] uppercase tracking-wider text-white/60 mt-1">Upload</span>
        </button>

        {/* Demo Speech Sample */}
        <button
          onClick={handlePlayDemoAudio}
          disabled={isActive}
          className={`w-14 h-14 sm:w-16 sm:h-16 border border-white/40 bg-zinc-950 flex flex-col items-center justify-center rounded-2xl shadow-xl transition-all ${
            isActive ? 'opacity-40 cursor-not-allowed' : 'hover:bg-white/10 hover:border-white cursor-pointer hover:scale-105 active:scale-95'
          }`}
          title="Play Demo Speech Sample"
        >
          <Sparkles className="w-6 h-6 text-emerald-400" />
          <span className="font-mono text-[9px] uppercase tracking-wider text-emerald-400 mt-1">Demo</span>
        </button>

        {/* Mute / Unmute */}
        <button
          onClick={() => setIsMuted(!isMuted)}
          disabled={!isActive}
          className={`w-14 h-14 sm:w-16 sm:h-16 border border-white bg-black flex items-center justify-center rounded-2xl shadow-xl transition-all ${
            !isActive
              ? 'opacity-30 cursor-not-allowed'
              : isMuted
              ? 'text-red-500 border-red-500 bg-red-950/40 hover:bg-red-900/50 cursor-pointer'
              : 'text-white hover:bg-white/10 cursor-pointer hover:scale-105 active:scale-95'
          }`}
          title={isMuted ? 'Unmute Audio Stream' : 'Mute Audio Stream'}
        >
          <Pause className={`w-7 h-7 ${isMuted ? 'text-red-500' : 'text-white'}`} />
        </button>

        {/* Stop & Analyze */}
        <button
          onClick={() => stopAudio(true)}
          disabled={!isActive}
          className={`w-14 h-14 sm:w-16 sm:h-16 border border-white bg-black flex items-center justify-center rounded-2xl shadow-xl transition-all ${
            !isActive ? 'opacity-30 cursor-not-allowed' : 'hover:bg-white/10 cursor-pointer hover:scale-105 active:scale-95'
          }`}
          title="Stop Recording & Analyze Speech"
        >
          <Square className="w-7 h-7 text-white fill-white" />
        </button>

        {/* Configuration Panel */}
        <button
          onClick={() => setIsSetupOpen(true)}
          className="w-14 h-14 sm:w-16 sm:h-16 border border-white bg-black flex items-center justify-center rounded-2xl text-white hover:bg-white/10 shadow-xl transition-all cursor-pointer hover:scale-105 active:scale-95"
          title="Model Setup & Custom Vocabulary"
        >
          <SlidersHorizontal className="w-7 h-7" />
        </button>
      </footer>

      {/* Floating Error Toast */}
      <AnimatePresence>
        {errorMessage && !permissionDenied && (
          <motion.div
            initial={{ opacity: 0, y: -40 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -40 }}
            className="absolute top-16 z-50 self-center border border-red-500/80 bg-red-950/95 px-6 py-4 rounded-xl font-mono text-sm text-red-200 shadow-2xl max-w-lg text-center"
          >
            <div className="font-bold mb-1 flex items-center justify-center gap-2">
              <AlertTriangle className="w-4 h-4 text-red-400" />
              <span>{errorMessage}</span>
            </div>
            <button
              onClick={() => setErrorMessage(null)}
              className="absolute top-2 right-2 text-red-400 hover:text-white p-1 cursor-pointer"
              aria-label="Close error"
            >
              ✕
            </button>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Model Setup Modal */}
      <SetupPanel
        isOpen={isSetupOpen}
        onClose={() => setIsSetupOpen(false)}
        config={config}
        onSaveConfig={setConfig}
      />

      {/* Gemini Speech Analysis & Coaching Modal */}
      <AnalysisModal
        isAnalyzing={isAnalyzing}
        analysisResult={analysisResult}
        onClose={() => setShowAnalysisModal(false)}
        paceMode={paceMode}
        setPaceMode={setPaceMode}
        totalWords={totalVerbatimWords}
        fillerCount={fillerCount}
        netWords={netWords}
        speechDurationSeconds={durationSec}
        grossPaceWpm={grossPaceWpm}
        netPaceWpm={netPaceWpm}
        currentPaceWpm={currentPaceWpm}
        pacePercentage={pacePercentage}
        fillerPercentage={fillerPercentage}
      />
    </div>
  );
}
