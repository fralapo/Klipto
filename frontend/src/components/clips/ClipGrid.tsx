import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, ChevronLeft, ChevronRight, Download, Share2 } from 'lucide-react';
import ClipCard from './ClipCard';
import { Button } from '../ui';
import { VideoClip } from '../../types';
import { downloadClip, getClipUrl } from '../../lib/api';

interface ClipGridProps {
  clips: VideoClip[];
  onDeleteClip?: (id: number) => void;
}

export default function ClipGrid({ clips, onDeleteClip }: ClipGridProps) {
  const [previewClip, setPreviewClip] = useState<VideoClip | null>(null);
  const [previewIndex, setPreviewIndex] = useState(0);

  const openPreview = (clip: VideoClip) => {
    const index = clips.findIndex(c => c.id === clip.id);
    setPreviewIndex(index);
    setPreviewClip(clip);
  };

  const closePreview = () => {
    setPreviewClip(null);
  };

  const nextClip = () => {
    const newIndex = (previewIndex + 1) % clips.length;
    setPreviewIndex(newIndex);
    setPreviewClip(clips[newIndex]);
  };

  const prevClip = () => {
    const newIndex = (previewIndex - 1 + clips.length) % clips.length;
    setPreviewIndex(newIndex);
    setPreviewClip(clips[newIndex]);
  };

  if (clips.length === 0) {
    return (
      <div className="text-center py-16">
        <p className="text-dark-400">Nessun clip generato ancora.</p>
      </div>
    );
  }

  return (
    <>
      {/* Grid */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-6"
      >
        {clips.map((clip, index) => (
          <ClipCard
            key={clip.id}
            clip={clip}
            index={index}
            onDelete={onDeleteClip}
            onPreview={openPreview}
          />
        ))}
      </motion.div>

      {/* Fullscreen Preview Modal */}
      <AnimatePresence>
        {previewClip && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/95 backdrop-blur-xl"
            onClick={closePreview}
          >
            {/* Close button */}
            <button
              onClick={closePreview}
              className="absolute top-4 right-4 p-3 rounded-full bg-white/10 hover:bg-white/20 transition-colors z-10"
            >
              <X className="w-6 h-6 text-white" />
            </button>

            {/* Navigation buttons */}
            {clips.length > 1 && (
              <>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    prevClip();
                  }}
                  className="absolute left-4 top-1/2 -translate-y-1/2 p-3 rounded-full bg-white/10 hover:bg-white/20 transition-colors z-10"
                >
                  <ChevronLeft className="w-6 h-6 text-white" />
                </button>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    nextClip();
                  }}
                  className="absolute right-4 top-1/2 -translate-y-1/2 p-3 rounded-full bg-white/10 hover:bg-white/20 transition-colors z-10"
                >
                  <ChevronRight className="w-6 h-6 text-white" />
                </button>
              </>
            )}

            {/* Video Container */}
            <motion.div
              key={previewClip.id}
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.9, opacity: 0 }}
              className="relative max-w-sm w-full mx-4"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="aspect-[9/16] rounded-2xl overflow-hidden bg-dark-900 shadow-2xl">
                <video
                  src={previewClip.url || getClipUrl(previewClip.path)}
                  className="w-full h-full object-cover"
                  controls
                  autoPlay
                  loop
                />
              </div>

              {/* Info panel */}
              <div className="mt-4 glass rounded-xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-white font-semibold">Clip #{previewClip.id}</span>
                  <span className="text-primary-400 font-bold">Score: {previewClip.score}</span>
                </div>
                <p className="text-dark-300 text-sm mb-4">{previewClip.reason}</p>

                <div className="flex gap-2">
                  <Button
                    variant="primary"
                    size="sm"
                    icon={<Download className="w-4 h-4" />}
                    onClick={() => downloadClip(previewClip.path)}
                    className="flex-1"
                  >
                    Download
                  </Button>
                  <Button
                    variant="secondary"
                    size="sm"
                    icon={<Share2 className="w-4 h-4" />}
                    onClick={() => {
                      navigator.clipboard.writeText(previewClip.url || getClipUrl(previewClip.path));
                    }}
                  >
                    Copia Link
                  </Button>
                </div>
              </div>

              {/* Clip counter */}
              <div className="absolute -bottom-12 left-1/2 -translate-x-1/2 flex gap-2">
                {clips.map((_, idx) => (
                  <button
                    key={idx}
                    onClick={(e) => {
                      e.stopPropagation();
                      setPreviewIndex(idx);
                      setPreviewClip(clips[idx]);
                    }}
                    className={`w-2 h-2 rounded-full transition-all ${
                      idx === previewIndex
                        ? 'bg-primary-500 w-6'
                        : 'bg-white/30 hover:bg-white/50'
                    }`}
                  />
                ))}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
