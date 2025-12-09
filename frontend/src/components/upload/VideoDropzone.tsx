import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';
import { Upload, Film, X, Sparkles } from 'lucide-react';
import { Button, Card, Progress } from '../ui';

interface VideoDropzoneProps {
  onUpload: (file: File, numClips: number) => void;
  isUploading: boolean;
  uploadProgress: number;
}

export default function VideoDropzone({ onUpload, isUploading, uploadProgress }: VideoDropzoneProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [numClips, setNumClips] = useState(3);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (file) {
      setSelectedFile(file);
      const url = URL.createObjectURL(file);
      setPreview(url);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/*': ['.mp4', '.mov', '.avi', '.mkv', '.webm'],
    },
    maxFiles: 1,
    disabled: isUploading,
  });

  const clearSelection = () => {
    if (preview) URL.revokeObjectURL(preview);
    setSelectedFile(null);
    setPreview(null);
  };

  const handleUpload = () => {
    if (selectedFile) {
      onUpload(selectedFile, numClips);
    }
  };

  return (
    <div className="w-full max-w-3xl mx-auto">
      <AnimatePresence mode="wait">
        {!selectedFile ? (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.3 }}
          >
            <div
              {...getRootProps()}
              className={`
                relative overflow-hidden rounded-3xl border-2 border-dashed
                transition-all duration-300 cursor-pointer
                ${isDragActive
                  ? 'border-primary-500 bg-primary-500/10 scale-[1.02]'
                  : 'border-dark-600 hover:border-primary-500/50 hover:bg-white/5'
                }
              `}
            >
              <input {...getInputProps()} />

              {/* Background gradient effect */}
              <div className="absolute inset-0 bg-gradient-to-br from-primary-500/5 via-transparent to-accent-500/5" />

              <div className="relative p-12 flex flex-col items-center justify-center min-h-[400px]">
                <motion.div
                  animate={{
                    y: isDragActive ? -10 : 0,
                    scale: isDragActive ? 1.1 : 1,
                  }}
                  transition={{ type: 'spring', stiffness: 300 }}
                  className="relative"
                >
                  <div className="absolute inset-0 bg-gradient-to-r from-primary-500 to-accent-500 rounded-full blur-2xl opacity-30" />
                  <div className="relative w-24 h-24 rounded-full bg-gradient-to-r from-primary-500 to-accent-500 flex items-center justify-center">
                    <Upload className="w-10 h-10 text-white" />
                  </div>
                </motion.div>

                <h3 className="mt-8 text-2xl font-bold text-white">
                  {isDragActive ? 'Rilascia il video qui' : 'Carica il tuo video'}
                </h3>
                <p className="mt-3 text-dark-400 text-center max-w-md">
                  Trascina un video o clicca per selezionare. Supporta MP4, MOV, AVI e altri formati.
                </p>

                <div className="mt-6 flex items-center gap-4 text-sm text-dark-500">
                  <span className="flex items-center gap-1">
                    <Film className="w-4 h-4" />
                    Video orizzontali
                  </span>
                  <span>•</span>
                  <span>Max 500MB</span>
                  <span>•</span>
                  <span>Output 9:16</span>
                </div>
              </div>
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.95 }}
            transition={{ duration: 0.3 }}
          >
            <Card variant="default" className="relative">
              {!isUploading && (
                <button
                  onClick={clearSelection}
                  className="absolute top-4 right-4 p-2 rounded-full bg-dark-800 hover:bg-dark-700 transition-colors z-10"
                >
                  <X className="w-5 h-5 text-dark-400" />
                </button>
              )}

              <div className="grid md:grid-cols-2 gap-6">
                {/* Video Preview */}
                <div className="relative rounded-xl overflow-hidden bg-dark-900 aspect-video">
                  {preview && (
                    <video
                      src={preview}
                      className="w-full h-full object-cover"
                      controls
                      muted
                    />
                  )}
                </div>

                {/* Upload Settings */}
                <div className="flex flex-col justify-between">
                  <div>
                    <h3 className="text-xl font-bold text-white mb-2 truncate">
                      {selectedFile.name}
                    </h3>
                    <p className="text-dark-400 text-sm mb-6">
                      {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
                    </p>

                    {/* Number of clips selector */}
                    <div className="mb-6">
                      <label className="block text-sm font-medium text-dark-300 mb-3">
                        Numero di clip da generare
                      </label>
                      <div className="flex gap-3">
                        {[1, 2, 3, 5, 10].map((num) => (
                          <button
                            key={num}
                            onClick={() => setNumClips(num)}
                            disabled={isUploading}
                            className={`
                              w-12 h-12 rounded-xl font-semibold transition-all duration-300
                              ${numClips === num
                                ? 'bg-gradient-to-r from-primary-500 to-accent-500 text-white scale-110'
                                : 'bg-dark-800 text-dark-400 hover:bg-dark-700 hover:text-white'
                              }
                              disabled:opacity-50 disabled:cursor-not-allowed
                            `}
                          >
                            {num}
                          </button>
                        ))}
                      </div>
                    </div>

                    {/* Upload Progress */}
                    {isUploading && (
                      <div className="mb-6">
                        <Progress value={uploadProgress} label="Caricamento video" />
                      </div>
                    )}
                  </div>

                  <Button
                    onClick={handleUpload}
                    disabled={isUploading}
                    isLoading={isUploading}
                    icon={<Sparkles className="w-5 h-5" />}
                    className="w-full"
                    size="lg"
                  >
                    {isUploading ? 'Caricamento...' : 'Genera Clip Virali'}
                  </Button>
                </div>
              </div>
            </Card>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
