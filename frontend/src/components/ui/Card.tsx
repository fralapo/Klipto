import { ReactNode } from 'react';
import { motion } from 'framer-motion';

interface CardProps {
  className?: string;
  variant?: 'default' | 'gradient' | 'glow';
  hover?: boolean;
  children: ReactNode;
  onClick?: () => void;
}

export default function Card({ className = '', variant = 'default', hover = false, children, onClick }: CardProps) {
  const baseStyles = 'rounded-2xl p-6 transition-all duration-300';

  const variants = {
    default: 'glass',
    gradient: 'gradient-border',
    glow: 'glass glow',
  };

  const hoverStyles = hover ? 'hover:bg-white/10 hover:scale-[1.02] cursor-pointer' : '';

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={`${baseStyles} ${variants[variant]} ${hoverStyles} ${className}`}
      onClick={onClick}
    >
      {children}
    </motion.div>
  );
}
