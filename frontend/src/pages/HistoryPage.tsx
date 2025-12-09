import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Clock,
  Film,
  Trash2,
  ChevronRight,
  Calendar,
  CheckCircle,
  XCircle,
  Loader2
} from 'lucide-react';
import { Card, Button, Badge } from '../components/ui';

interface HistoryItem {
  id: string;
  filename: string;
  date: string;
  status: 'completed' | 'failed' | 'processing';
  clipsCount: number;
  duration: string;
}

// Mock data - in production this would come from the API
const mockHistory: HistoryItem[] = [
  {
    id: '1',
    filename: 'podcast_episode_45.mp4',
    date: '2024-01-15T10:30:00',
    status: 'completed',
    clipsCount: 5,
    duration: '45:30',
  },
  {
    id: '2',
    filename: 'interview_marketing.mov',
    date: '2024-01-14T15:20:00',
    status: 'completed',
    clipsCount: 3,
    duration: '28:15',
  },
  {
    id: '3',
    filename: 'tutorial_react.mp4',
    date: '2024-01-14T09:00:00',
    status: 'failed',
    clipsCount: 0,
    duration: '15:00',
  },
];

export default function HistoryPage() {
  const [history, setHistory] = useState<HistoryItem[]>(mockHistory);
  const [selectedItem, setSelectedItem] = useState<string | null>(null);

  const handleDelete = (id: string) => {
    setHistory(history.filter(item => item.id !== id));
  };

  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return new Intl.DateTimeFormat('it-IT', {
      day: 'numeric',
      month: 'short',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    }).format(date);
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'failed':
        return <XCircle className="w-5 h-5 text-red-400" />;
      case 'processing':
        return <Loader2 className="w-5 h-5 text-primary-400 animate-spin" />;
      default:
        return null;
    }
  };

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'completed':
        return <Badge variant="success">Completato</Badge>;
      case 'failed':
        return <Badge variant="danger">Fallito</Badge>;
      case 'processing':
        return <Badge variant="info">In Elaborazione</Badge>;
      default:
        return null;
    }
  };

  return (
    <div className="container mx-auto px-4 max-w-4xl">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div>
            <h1 className="text-3xl font-bold text-white mb-2">Cronologia</h1>
            <p className="text-dark-400">
              Visualizza i video elaborati in precedenza
            </p>
          </div>
          {history.length > 0 && (
            <Button
              variant="ghost"
              onClick={() => setHistory([])}
              icon={<Trash2 className="w-4 h-4" />}
              className="text-red-400 hover:text-red-300"
            >
              Cancella tutto
            </Button>
          )}
        </div>

        {/* Empty State */}
        {history.length === 0 ? (
          <Card className="text-center py-16">
            <div className="w-16 h-16 rounded-full bg-dark-800 flex items-center justify-center mx-auto mb-4">
              <Clock className="w-8 h-8 text-dark-500" />
            </div>
            <h3 className="text-xl font-semibold text-white mb-2">
              Nessun video nella cronologia
            </h3>
            <p className="text-dark-400 mb-6">
              I video che elabori appariranno qui
            </p>
            <Button variant="primary" onClick={() => window.location.href = '/'}>
              Carica il primo video
            </Button>
          </Card>
        ) : (
          <div className="space-y-4">
            {history.map((item, index) => (
              <motion.div
                key={item.id}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.05 }}
              >
                <Card
                  hover
                  className={`
                    cursor-pointer transition-all duration-300
                    ${selectedItem === item.id ? 'ring-2 ring-primary-500/50' : ''}
                  `}
                  onClick={() => setSelectedItem(selectedItem === item.id ? null : item.id)}
                >
                  <div className="flex items-center gap-4">
                    {/* Status Icon */}
                    <div className="flex-shrink-0">
                      {getStatusIcon(item.status)}
                    </div>

                    {/* Video Icon */}
                    <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-primary-500/20 to-accent-500/20 flex items-center justify-center flex-shrink-0">
                      <Film className="w-6 h-6 text-primary-400" />
                    </div>

                    {/* Info */}
                    <div className="flex-1 min-w-0">
                      <h3 className="text-white font-semibold truncate">
                        {item.filename}
                      </h3>
                      <div className="flex items-center gap-4 mt-1 text-sm text-dark-400">
                        <span className="flex items-center gap-1">
                          <Calendar className="w-4 h-4" />
                          {formatDate(item.date)}
                        </span>
                        <span className="flex items-center gap-1">
                          <Clock className="w-4 h-4" />
                          {item.duration}
                        </span>
                      </div>
                    </div>

                    {/* Status & Actions */}
                    <div className="flex items-center gap-4 flex-shrink-0">
                      {item.status === 'completed' && (
                        <div className="text-right">
                          <div className="text-2xl font-bold text-white">{item.clipsCount}</div>
                          <div className="text-xs text-dark-400">clip</div>
                        </div>
                      )}
                      {getStatusBadge(item.status)}
                      <ChevronRight className={`w-5 h-5 text-dark-500 transition-transform ${selectedItem === item.id ? 'rotate-90' : ''}`} />
                    </div>
                  </div>

                  {/* Expanded Content */}
                  {selectedItem === item.id && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      className="mt-4 pt-4 border-t border-dark-700"
                    >
                      <div className="flex items-center justify-between">
                        <div className="text-sm text-dark-400">
                          {item.status === 'completed'
                            ? `Generati ${item.clipsCount} clip virali dal video`
                            : item.status === 'failed'
                            ? 'Elaborazione fallita. Prova a ricaricare il video.'
                            : 'Elaborazione in corso...'}
                        </div>
                        <div className="flex gap-2">
                          {item.status === 'completed' && (
                            <Button variant="primary" size="sm">
                              Visualizza Clip
                            </Button>
                          )}
                          {item.status === 'failed' && (
                            <Button variant="secondary" size="sm">
                              Riprova
                            </Button>
                          )}
                          <Button
                            variant="ghost"
                            size="sm"
                            icon={<Trash2 className="w-4 h-4" />}
                            onClick={(e) => {
                              e.stopPropagation();
                              handleDelete(item.id);
                            }}
                            className="text-red-400 hover:text-red-300"
                          >
                            Elimina
                          </Button>
                        </div>
                      </div>
                    </motion.div>
                  )}
                </Card>
              </motion.div>
            ))}
          </div>
        )}
      </motion.div>
    </div>
  );
}
