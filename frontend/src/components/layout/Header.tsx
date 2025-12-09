import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Scissors, Settings, History, Home, Github } from 'lucide-react';

export default function Header() {
  const location = useLocation();

  const navItems = [
    { path: '/', label: 'Home', icon: Home },
    { path: '/history', label: 'Cronologia', icon: History },
    { path: '/settings', label: 'Impostazioni', icon: Settings },
  ];

  return (
    <header className="fixed top-0 left-0 right-0 z-40 glass-strong">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-3 group">
            <motion.div
              whileHover={{ rotate: 15 }}
              className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center"
            >
              <Scissors className="w-5 h-5 text-white" />
            </motion.div>
            <div>
              <h1 className="text-xl font-bold text-gradient">Klipto</h1>
              <p className="text-xs text-dark-400 -mt-0.5">AI Video Repurposing</p>
            </div>
          </Link>

          {/* Navigation */}
          <nav className="hidden md:flex items-center gap-1">
            {navItems.map((item) => {
              const isActive = location.pathname === item.path;
              const Icon = item.icon;

              return (
                <Link
                  key={item.path}
                  to={item.path}
                  className={`
                    relative px-4 py-2 rounded-xl flex items-center gap-2
                    transition-all duration-300
                    ${isActive
                      ? 'text-white'
                      : 'text-dark-400 hover:text-white hover:bg-white/5'
                    }
                  `}
                >
                  {isActive && (
                    <motion.div
                      layoutId="nav-active"
                      className="absolute inset-0 bg-white/10 rounded-xl"
                      transition={{ type: 'spring', duration: 0.5 }}
                    />
                  )}
                  <Icon className="w-4 h-4 relative z-10" />
                  <span className="relative z-10 font-medium">{item.label}</span>
                </Link>
              );
            })}
          </nav>

          {/* GitHub Link */}
          <a
            href="https://github.com"
            target="_blank"
            rel="noopener noreferrer"
            className="p-2 rounded-xl text-dark-400 hover:text-white hover:bg-white/5 transition-colors"
          >
            <Github className="w-5 h-5" />
          </a>
        </div>
      </div>
    </header>
  );
}
