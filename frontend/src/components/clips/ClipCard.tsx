import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Play,
  Download,
  Share2,
  Trash2,
  Star,
  Clock,
  Maximize2
} from 'lucide-react';
import { Button, Card, Badge } from '../ui';
import { VideoClip } from '../../types';
import { downloadClip, getClipUrl } from '../../lib/api';

interface ClipCardProps {
  clip: VideoClip;
  index: number;
  onDelete?: (id: number) => void;
  onPreview?: (clip: VideoClip) => void;
}

export default function ClipCard({ clip, index, onDelete, onPreview }: ClipCardProps) {
  const [isHovered, setIsHovered] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

  const clipUrl = clip.url || getClipUrl(clip.path);

  const handleDownload = () => {
    downloadClip(clip.path);
  };

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: `Clip ${clip.id}`,
          text: clip.reason,
          url: clipUrl,
        });
      } catch (err) {
        // User cancelled or error
      }
    } else {
      // Fallback: copy link to clipboard
      navigator.clipboard.writeText(clipUrl);
    }
  };

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-400';
    if (score >= 60) return 'text-yellow-400';
    return 'text-orange-400';
  };

  const getScoreBadge = (score: number) => {
    if (score >= 80) return 'success';
    if (score >= 60) return 'warning';
    return 'danger';
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.1 }}
      onHoverStart={() => setIsHovered(true)}
      onHoverEnd={() => setIsHovered(false)}
    >
      <Card className="overflow-hidden group" hover>
        {/* Video Container - 9:16 aspect ratio */}
        <div className="relative aspect-[9/16] rounded-xl overflow-hidden bg-dark-900 mb-4">
          <video
            src={clipUrl}
            className="absolute inset-0 w-full h-full object-cover"
            loop
            muted
            playsInline
            onMouseEnter={(e) => {
              e.currentTarget.play();
              setIsPlaying(true);
            }}
            onMouseLeave={(e) => {
              e.currentTarget.pause();
              e.currentTarget.currentTime = 0;
              setIsPlaying(false);
            }}
          />

          {/* Overlay */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: isHovered ? 1 : 0 }}
            className="absolute inset-0 bg-gradient-to-t from-black/80 via-transparent to-black/20"
          />

          {/* Play button overlay */}
          {!isPlaying && (
            <div className="absolute inset-0 flex items-center justify-center">
              <motion.div
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.95 }}
                className="w-16 h-16 rounded-full bg-white/20 backdrop-blur-sm flex items-center justify-center cursor-pointer"
                onClick={() => onPreview?.(clip)}
              >
                <Play className="w-8 h-8 text-white fill-white ml-1" />
              </motion.div>
            </div>
          )}

          {/* Top badges */}
          <div className="absolute top-3 left-3 flex gap-2">
            <Badge variant="info" size="sm">
              Clip #{clip.id}
            </Badge>
          </div>

          {/* Score badge */}
          <div className="absolute top-3 right-3">
            <div className={`flex items-center gap-1 px-2 py-1 rounded-full bg-black/50 backdrop-blur-sm ${getScoreColor(clip.score)}`}>
              <Star className="w-3 h-3 fill-current" />
              <span className="text-sm font-bold">{clip.score}</span>
            </div>
          </div>

          {/* Bottom info */}
          <motion.div
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: isHovered ? 0 : 20, opacity: isHovered ? 1 : 0 }}
            className="absolute bottom-3 left-3 right-3"
          >
            <div className="flex items-center gap-2 text-white/80 text-sm">
              <Clock className="w-4 h-4" />
              <span>{clip.duration?.toFixed(1)}s</span>
            </div>
          </motion.div>

          {/* Fullscreen button */}
          <motion.button
            initial={{ opacity: 0 }}
            animate={{ opacity: isHovered ? 1 : 0 }}
            className="absolute bottom-3 right-3 p-2 rounded-lg bg-black/50 backdrop-blur-sm text-white hover:bg-white/20 transition-colors"
            onClick={() => onPreview?.(clip)}
          >
            <Maximize2 className="w-4 h-4" />
          </motion.button>
        </div>

        {/* Content */}
        <div className="space-y-4">
          {/* Reason/Description */}
          <div>
            <p className="text-dark-300 text-sm line-clamp-2">
              {clip.reason || 'Clip estratta automaticamente'}
            </p>
          </div>

          {/* Score Bar */}
          <div className="space-y-1">
            <div className="flex justify-between items-center text-xs">
              <span className="text-dark-400">Virality Score</span>
              <Badge variant={getScoreBadge(clip.score)} size="sm">
                {clip.score >= 80 ? 'Alta' : clip.score >= 60 ? 'Media' : 'Bassa'}
              </Badge>
            </div>
            <div className="h-1.5 bg-dark-800 rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${clip.score}%` }}
                transition={{ duration: 0.8, delay: index * 0.1 }}
                className={`h-full rounded-full ${
                  clip.score >= 80
                    ? 'bg-gradient-to-r from-green-500 to-emerald-400'
                    : clip.score >= 60
                    ? 'bg-gradient-to-r from-yellow-500 to-amber-400'
                    : 'bg-gradient-to-r from-orange-500 to-red-400'
                }`}
              />
            </div>
          </div>

          {/* Actions */}
          <div className="flex gap-2 pt-2">
            <Button
              variant="primary"
              size="sm"
              icon={<Download className="w-4 h-4" />}
              onClick={handleDownload}
              className="flex-1"
            >
              Download
            </Button>
            <Button
              variant="secondary"
              size="sm"
              icon={<Share2 className="w-4 h-4" />}
              onClick={handleShare}
            />
            {onDelete && (
              <Button
                variant="ghost"
                size="sm"
                icon={<Trash2 className="w-4 h-4" />}
                onClick={() => onDelete(clip.id)}
                className="text-red-400 hover:text-red-300 hover:bg-red-500/10"
              />
            )}
          </div>
        </div>
      </Card>
    </motion.div>
  );
}
