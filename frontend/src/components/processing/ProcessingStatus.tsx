import { motion } from 'framer-motion';
import {
  Mic,
  Film,
  Brain,
  Scissors,
  CheckCircle,
  AlertCircle,
  Loader2
} from 'lucide-react';
import { Card, Progress, Badge } from '../ui';
import { ProcessingTask } from '../../types';

interface ProcessingStatusProps {
  task: ProcessingTask | null;
  error: string | null;
}

const stages = [
  { key: 'Initializing', icon: Loader2, label: 'Inizializzazione' },
  { key: 'Extracting Audio', icon: Mic, label: 'Estrazione Audio' },
  { key: 'Transcribing (Whisper)', icon: Mic, label: 'Trascrizione (Whisper)' },
  { key: 'Detecting Scenes', icon: Film, label: 'Rilevamento Scene' },
  { key: 'Analyzing Content', icon: Brain, label: 'Analisi Contenuto (AI)' },
  { key: 'Rendering Clips', icon: Scissors, label: 'Rendering Clip' },
];

export default function ProcessingStatus({ task, error }: ProcessingStatusProps) {
  if (!task) return null;

  const currentStage = task.progress?.stage || '';
  const currentProgress = task.progress?.progress || 0;
  const isComplete = task.status === 'SUCCESS';
  const isFailed = task.status === 'FAILURE';

  const getStageStatus = (stageKey: string) => {
    const stageIndex = stages.findIndex(s => s.key === stageKey);
    const currentIndex = stages.findIndex(s => s.key === currentStage);

    if (isComplete) return 'complete';
    if (isFailed) return 'error';
    if (stageIndex < currentIndex) return 'complete';
    if (stageIndex === currentIndex) return 'active';
    return 'pending';
  };

  return (
    <Card className="w-full max-w-3xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-xl font-bold text-white">Elaborazione Video</h3>
          <p className="text-dark-400 text-sm mt-1">
            {isComplete ? 'Completato!' : isFailed ? 'Errore durante l\'elaborazione' : 'In corso...'}
          </p>
        </div>
        <Badge variant={isComplete ? 'success' : isFailed ? 'danger' : 'info'} size="md">
          {isComplete ? 'Completato' : isFailed ? 'Errore' : 'In Elaborazione'}
        </Badge>
      </div>

      {/* Overall Progress */}
      <div className="mb-8">
        <Progress
          value={isComplete ? 100 : currentProgress}
          label="Progresso totale"
          size="lg"
        />
      </div>

      {/* Stage Timeline */}
      <div className="relative">
        {/* Connecting Line */}
        <div className="absolute left-5 top-0 bottom-0 w-0.5 bg-dark-700" />

        <div className="space-y-4">
          {stages.map((stage, index) => {
            const status = getStageStatus(stage.key);
            const Icon = stage.icon;

            return (
              <motion.div
                key={stage.key}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="relative flex items-center gap-4"
              >
                {/* Icon Circle */}
                <div
                  className={`
                    relative z-10 w-10 h-10 rounded-full flex items-center justify-center
                    transition-all duration-300
                    ${status === 'complete'
                      ? 'bg-green-500'
                      : status === 'active'
                      ? 'bg-gradient-to-r from-primary-500 to-accent-500'
                      : status === 'error'
                      ? 'bg-red-500'
                      : 'bg-dark-700'
                    }
                  `}
                >
                  {status === 'complete' ? (
                    <CheckCircle className="w-5 h-5 text-white" />
                  ) : status === 'active' ? (
                    <Icon className="w-5 h-5 text-white animate-pulse" />
                  ) : status === 'error' ? (
                    <AlertCircle className="w-5 h-5 text-white" />
                  ) : (
                    <Icon className="w-5 h-5 text-dark-400" />
                  )}

                  {/* Pulse effect for active */}
                  {status === 'active' && (
                    <div className="absolute inset-0 rounded-full bg-primary-500 animate-ping opacity-30" />
                  )}
                </div>

                {/* Label */}
                <div className="flex-1">
                  <span
                    className={`
                      font-medium transition-colors duration-300
                      ${status === 'complete'
                        ? 'text-green-400'
                        : status === 'active'
                        ? 'text-white'
                        : status === 'error'
                        ? 'text-red-400'
                        : 'text-dark-500'
                      }
                    `}
                  >
                    {stage.label}
                  </span>
                </div>

                {/* Status indicator */}
                {status === 'active' && (
                  <Loader2 className="w-5 h-5 text-primary-400 animate-spin" />
                )}
              </motion.div>
            );
          })}
        </div>
      </div>

      {/* Error Message */}
      {(error || isFailed) && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-6 p-4 rounded-xl bg-red-500/10 border border-red-500/30"
        >
          <div className="flex items-start gap-3">
            <AlertCircle className="w-5 h-5 text-red-400 flex-shrink-0 mt-0.5" />
            <div>
              <p className="text-red-400 font-medium">Errore durante l'elaborazione</p>
              <p className="text-dark-400 text-sm mt-1">
                {error || task.error || 'Si è verificato un errore. Riprova.'}
              </p>
            </div>
          </div>
        </motion.div>
      )}
    </Card>
  );
}
