import { motion } from 'framer-motion';

interface ProgressProps {
  value: number;
  max?: number;
  showLabel?: boolean;
  size?: 'sm' | 'md' | 'lg';
  label?: string;
}

export default function Progress({
  value,
  max = 100,
  showLabel = true,
  size = 'md',
  label
}: ProgressProps) {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100);

  const sizes = {
    sm: 'h-1.5',
    md: 'h-2.5',
    lg: 'h-4',
  };

  return (
    <div className="w-full">
      {(showLabel || label) && (
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm text-dark-300">{label || 'Progress'}</span>
          <span className="text-sm font-medium text-white">{Math.round(percentage)}%</span>
        </div>
      )}
      <div className={`w-full bg-dark-800 rounded-full overflow-hidden ${sizes[size]}`}>
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${percentage}%` }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          className={`${sizes[size]} rounded-full bg-gradient-to-r from-primary-500 to-accent-500 relative`}
        >
          <div className="absolute inset-0 bg-white/20 animate-pulse" />
        </motion.div>
      </div>
    </div>
  );
}
