import { ReactNode } from 'react';
import Header from './Header';

interface LayoutProps {
  children: ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  return (
    <div className="min-h-screen bg-dark-950 relative overflow-hidden">
      {/* Background effects */}
      <div className="fixed inset-0 pointer-events-none">
        {/* Gradient orbs */}
        <div className="absolute top-0 -left-40 w-96 h-96 bg-primary-500/20 rounded-full blur-[120px]" />
        <div className="absolute top-1/3 -right-40 w-96 h-96 bg-accent-500/20 rounded-full blur-[120px]" />
        <div className="absolute bottom-0 left-1/3 w-96 h-96 bg-primary-600/10 rounded-full blur-[120px]" />

        {/* Grid pattern */}
        <div 
          className="absolute inset-0 opacity-[0.03]"
          style={{
            backgroundImage: `
              linear-gradient(rgba(255,255,255,0.1) 1px, transparent 1px),
              linear-gradient(90deg, rgba(255,255,255,0.1) 1px, transparent 1px)
            `,
            backgroundSize: '100px 100px',
          }}
        />
      </div>

      {/* Header */}
      <Header />

      {/* Main Content */}
      <main className="relative z-10 pt-20 pb-12">
        {children}
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/5">
        <div className="container mx-auto px-4 py-6">
          <div className="flex flex-col md:flex-row items-center justify-between gap-4">
            <p className="text-dark-500 text-sm">
              © 2024 Klipto. AI-powered video repurposing.
            </p>
            <div className="flex items-center gap-6">
              <a href="#" className="text-dark-500 hover:text-dark-300 text-sm transition-colors">
                Privacy
              </a>
              <a href="#" className="text-dark-500 hover:text-dark-300 text-sm transition-colors">
                Termini
              </a>
              <a href="#" className="text-dark-500 hover:text-dark-300 text-sm transition-colors">
                Documentazione
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}
