import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Sparkles, Zap, Shield, Cpu, ArrowRight, RefreshCw } from 'lucide-react';
import VideoDropzone from '../components/upload/VideoDropzone';
import ProcessingStatus from '../components/processing/ProcessingStatus';
import ClipGrid from '../components/clips/ClipGrid';
import { Card, Button } from '../components/ui';
import { uploadVideo } from '../lib/api';
import { useTaskStatus } from '../hooks/useTaskStatus';
import { VideoClip } from '../types';

type AppState = 'upload' | 'processing' | 'results';

export default function HomePage() {
  const [appState, setAppState] = useState<AppState>('upload');
  const [taskId, setTaskId] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [clips, setClips] = useState<VideoClip[]>([]);

  const { task, error } = useTaskStatus(taskId, {
    enabled: appState === 'processing',
  });

  // Check if processing is complete
  if (task?.status === 'SUCCESS' && appState === 'processing') {
    setClips(task.result?.clips || []);
    setAppState('results');
  }

  const handleUpload = useCallback(async (file: File, numClips: number) => {
    setIsUploading(true);
    setUploadProgress(0);

    try {
      const response = await uploadVideo(file, numClips, (progress) => {
        setUploadProgress(progress);
      });

      setTaskId(response.task_id);
      setAppState('processing');
    } catch (err) {
      console.error('Upload failed:', err);
    } finally {
      setIsUploading(false);
    }
  }, []);

  const handleReset = () => {
    setAppState('upload');
    setTaskId(null);
    setClips([]);
    setUploadProgress(0);
  };

  const features = [
    {
      icon: Zap,
      title: 'Elaborazione Veloce',
      description: 'Pipeline ottimizzata con AI locale per risultati rapidi',
    },
    {
      icon: Shield,
      title: 'Privacy First',
      description: 'I tuoi video restano sul tuo server, nessun cloud esterno',
    },
    {
      icon: Cpu,
      title: 'AI Avanzata',
      description: 'YOLOv8 + Whisper + LLM per clip intelligenti',
    },
  ];

  return (
    <div className="container mx-auto px-4">
      <AnimatePresence mode="wait">
        {appState === 'upload' && (
          <motion.div
            key="upload"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
          >
            {/* Hero Section */}
            <div className="text-center max-w-4xl mx-auto mb-12">
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.1 }}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-500/10 border border-primary-500/30 text-primary-400 text-sm font-medium mb-6"
              >
                <Sparkles className="w-4 h-4" />
                AI-Powered Video Repurposing
              </motion.div>

              <motion.h1
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="text-4xl md:text-6xl font-bold mb-6"
              >
                Trasforma i tuoi video in{' '}
                <span className="text-gradient">Shorts Virali</span>
              </motion.h1>

              <motion.p
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 }}
                className="text-lg text-dark-400 max-w-2xl mx-auto"
              >
                Carica un video orizzontale e lascia che l'AI identifichi i momenti più 
                coinvolgenti, tagliandoli automaticamente in formato 9:16 per TikTok, 
                Instagram Reels e YouTube Shorts.
              </motion.p>
            </div>

            {/* Upload Zone */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              <VideoDropzone
                onUpload={handleUpload}
                isUploading={isUploading}
                uploadProgress={uploadProgress}
              />
            </motion.div>

            {/* Features */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="mt-16 grid md:grid-cols-3 gap-6 max-w-4xl mx-auto"
            >
              {features.map((feature, index) => (
                <Card key={index} className="text-center" hover>
                  <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary-500/20 to-accent-500/20 flex items-center justify-center mx-auto mb-4">
                    <feature.icon className="w-6 h-6 text-primary-400" />
                  </div>
                  <h3 className="text-lg font-semibold text-white mb-2">
                    {feature.title}
                  </h3>
                  <p className="text-dark-400 text-sm">
                    {feature.description}
                  </p>
                </Card>
              ))}
            </motion.div>

            {/* How it works */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className="mt-20 text-center"
            >
              <h2 className="text-2xl font-bold text-white mb-8">Come Funziona</h2>
              <div className="flex flex-col md:flex-row items-center justify-center gap-4 md:gap-8">
                {[
                  { step: '1', title: 'Carica', desc: 'Upload del video' },
                  { step: '2', title: 'Analizza', desc: 'AI trova i momenti chiave' },
                  { step: '3', title: 'Genera', desc: 'Clip 9:16 automatici' },
                  { step: '4', title: 'Scarica', desc: 'Pubblica ovunque' },
                ].map((item, index) => (
                  <div key={index} className="flex items-center gap-4">
                    <div className="flex flex-col items-center">
                      <div className="w-12 h-12 rounded-full bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center text-white font-bold text-lg">
                        {item.step}
                      </div>
                      <h4 className="text-white font-semibold mt-2">{item.title}</h4>
                      <p className="text-dark-500 text-sm">{item.desc}</p>
                    </div>
                    {index < 3 && (
                      <ArrowRight className="w-5 h-5 text-dark-600 hidden md:block" />
                    )}
                  </div>
                ))}
              </div>
            </motion.div>
          </motion.div>
        )}

        {appState === 'processing' && (
          <motion.div
            key="processing"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="py-12"
          >
            <div className="text-center mb-8">
              <h2 className="text-3xl font-bold text-white mb-2">
                Elaborazione in Corso
              </h2>
              <p className="text-dark-400">
                L'AI sta analizzando il tuo video per trovare i momenti migliori
              </p>
            </div>

            <ProcessingStatus task={task} error={error} />

            {(task?.status === 'FAILURE' || error) && (
              <div className="mt-6 text-center">
                <Button onClick={handleReset} icon={<RefreshCw className="w-4 h-4" />}>
                  Riprova
                </Button>
              </div>
            )}
          </motion.div>
        )}

        {appState === 'results' && (
          <motion.div
            key="results"
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="py-8"
          >
            <div className="flex flex-col md:flex-row items-center justify-between gap-4 mb-8">
              <div>
                <h2 className="text-3xl font-bold text-white mb-2">
                  I Tuoi Clip Virali
                </h2>
                <p className="text-dark-400">
                  {clips.length} clip generati e pronti per il download
                </p>
              </div>
              <Button
                variant="secondary"
                onClick={handleReset}
                icon={<RefreshCw className="w-4 h-4" />}
              >
                Nuovo Video
              </Button>
            </div>

            <ClipGrid clips={clips} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
