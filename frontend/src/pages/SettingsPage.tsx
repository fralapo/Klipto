import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Save,
  Key,
  Brain,
  Mic,
  Server,
  CheckCircle,
  AlertCircle,
  Eye,
  EyeOff
} from 'lucide-react';
import { Card, Button, Badge } from '../components/ui';

interface Settings {
  llm_provider: string;
  llm_model: string;
  api_key: string;
  whisper_model: string;
  use_local_whisper: boolean;
}

const llmProviders = [
  { id: 'openrouter', name: 'OpenRouter', description: 'DeepSeek, Claude, GPT e altri', models: ['deepseek/deepseek-r1', 'deepseek/deepseek-v3', 'anthropic/claude-3.5-sonnet'] },
  { id: 'openai', name: 'OpenAI', description: 'GPT-4, GPT-3.5', models: ['gpt-4-turbo', 'gpt-4', 'gpt-3.5-turbo'] },
  { id: 'anthropic', name: 'Anthropic', description: 'Claude 3.5, Claude 3', models: ['claude-3-5-sonnet-20241022', 'claude-3-opus-20240229'] },
];

const whisperModels = [
  { id: 'tiny', name: 'Tiny', description: 'Più veloce, meno preciso', vram: '~1GB' },
  { id: 'base', name: 'Base', description: 'Bilanciato', vram: '~1GB' },
  { id: 'small', name: 'Small', description: 'Buona precisione', vram: '~2GB' },
  { id: 'medium', name: 'Medium', description: 'Alta precisione', vram: '~5GB' },
  { id: 'large', name: 'Large', description: 'Massima precisione', vram: '~10GB' },
];

export default function SettingsPage() {
  const [settings, setSettings] = useState<Settings>({
    llm_provider: 'openrouter',
    llm_model: 'deepseek/deepseek-r1',
    api_key: '',
    whisper_model: 'base',
    use_local_whisper: true,
  });

  const [showApiKey, setShowApiKey] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [saveSuccess, setSaveSuccess] = useState(false);

  const handleSave = async () => {
    setIsSaving(true);
    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1000));
    setIsSaving(false);
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 3000);
  };

  const selectedProvider = llmProviders.find(p => p.id === settings.llm_provider);

  return (
    <div className="container mx-auto px-4 max-w-4xl">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-white mb-2">Impostazioni</h1>
          <p className="text-dark-400">
            Configura le API e i modelli AI per Klipto
          </p>
        </div>

        {/* Success notification */}
        {saveSuccess && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="mb-6 p-4 rounded-xl bg-green-500/10 border border-green-500/30 flex items-center gap-3"
          >
            <CheckCircle className="w-5 h-5 text-green-400" />
            <span className="text-green-400">Impostazioni salvate con successo!</span>
          </motion.div>
        )}

        <div className="space-y-6">
          {/* LLM Provider Section */}
          <Card>
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500/20 to-accent-500/20 flex items-center justify-center">
                <Brain className="w-5 h-5 text-primary-400" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">Provider LLM</h2>
                <p className="text-dark-400 text-sm">Per l'analisi dei contenuti e rilevamento hook</p>
              </div>
            </div>

            {/* Provider selection */}
            <div className="grid md:grid-cols-3 gap-4 mb-6">
              {llmProviders.map((provider) => (
                <button
                  key={provider.id}
                  onClick={() => setSettings({ ...settings, llm_provider: provider.id, llm_model: provider.models[0] })}
                  className={`
                    p-4 rounded-xl border text-left transition-all duration-300
                    ${settings.llm_provider === provider.id
                      ? 'bg-primary-500/10 border-primary-500/50 ring-1 ring-primary-500/30'
                      : 'bg-dark-800/50 border-dark-700 hover:border-dark-600'
                    }
                  `}
                >
                  <div className="font-semibold text-white mb-1">{provider.name}</div>
                  <div className="text-dark-400 text-sm">{provider.description}</div>
                </button>
              ))}
            </div>

            {/* Model selection */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-dark-300 mb-2">
                Modello
              </label>
              <select
                value={settings.llm_model}
                onChange={(e) => setSettings({ ...settings, llm_model: e.target.value })}
                className="input-field"
              >
                {selectedProvider?.models.map((model) => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
            </div>

            {/* API Key */}
            <div>
              <label className="block text-sm font-medium text-dark-300 mb-2">
                <div className="flex items-center gap-2">
                  <Key className="w-4 h-4" />
                  API Key
                </div>
              </label>
              <div className="relative">
                <input
                  type={showApiKey ? 'text' : 'password'}
                  value={settings.api_key}
                  onChange={(e) => setSettings({ ...settings, api_key: e.target.value })}
                  placeholder={`Inserisci la tua ${selectedProvider?.name} API Key`}
                  className="input-field pr-12"
                />
                <button
                  onClick={() => setShowApiKey(!showApiKey)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-dark-400 hover:text-white transition-colors"
                >
                  {showApiKey ? <EyeOff className="w-5 h-5" /> : <Eye className="w-5 h-5" />}
                </button>
              </div>
              <p className="mt-2 text-dark-500 text-sm">
                La chiave viene salvata solo localmente nel file .env
              </p>
            </div>
          </Card>

          {/* Whisper Section */}
          <Card>
            <div className="flex items-center gap-3 mb-6">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-accent-500/20 to-primary-500/20 flex items-center justify-center">
                <Mic className="w-5 h-5 text-accent-400" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">Trascrizione Audio</h2>
                <p className="text-dark-400 text-sm">Configurazione Faster Whisper</p>
              </div>
            </div>

            {/* Local whisper toggle */}
            <div className="flex items-center justify-between p-4 rounded-xl bg-dark-800/50 border border-dark-700 mb-6">
              <div>
                <div className="font-medium text-white">Usa Whisper Locale</div>
                <div className="text-dark-400 text-sm">Elabora l'audio sul tuo hardware</div>
              </div>
              <button
                onClick={() => setSettings({ ...settings, use_local_whisper: !settings.use_local_whisper })}
                className={`
                  relative w-14 h-8 rounded-full transition-colors duration-300
                  ${settings.use_local_whisper ? 'bg-primary-500' : 'bg-dark-700'}
                `}
              >
                <motion.div
                  animate={{ x: settings.use_local_whisper ? 24 : 4 }}
                  className="absolute top-1 w-6 h-6 rounded-full bg-white shadow-lg"
                />
              </button>
            </div>

            {/* Whisper model selection */}
            <div>
              <label className="block text-sm font-medium text-dark-300 mb-3">
                Modello Whisper
              </label>
              <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
                {whisperModels.map((model) => (
                  <button
                    key={model.id}
                    onClick={() => setSettings({ ...settings, whisper_model: model.id })}
                    disabled={!settings.use_local_whisper}
                    className={`
                      p-3 rounded-xl border text-center transition-all duration-300
                      ${!settings.use_local_whisper ? 'opacity-50 cursor-not-allowed' : ''}
                      ${settings.whisper_model === model.id
                        ? 'bg-accent-500/10 border-accent-500/50 ring-1 ring-accent-500/30'
                        : 'bg-dark-800/50 border-dark-700 hover:border-dark-600'
                      }
                    `}
                  >
                    <div className="font-semibold text-white text-sm">{model.name}</div>
                    <div className="text-dark-500 text-xs mt-1">{model.vram}</div>
                  </button>
                ))}
              </div>
              <p className="mt-3 text-dark-500 text-sm flex items-center gap-2">
                <AlertCircle className="w-4 h-4" />
                Modelli più grandi richiedono più VRAM/RAM
              </p>
            </div>
          </Card>

          {/* Backend Status */}
          <Card>
            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-green-500/20 to-emerald-500/20 flex items-center justify-center">
                <Server className="w-5 h-5 text-green-400" />
              </div>
              <div>
                <h2 className="text-lg font-semibold text-white">Stato Backend</h2>
                <p className="text-dark-400 text-sm">Connessione ai servizi</p>
              </div>
            </div>

            <div className="space-y-3">
              {[
                { name: 'FastAPI Server', status: 'online' },
                { name: 'Redis', status: 'online' },
                { name: 'Celery Worker', status: 'online' },
              ].map((service) => (
                <div key={service.name} className="flex items-center justify-between p-3 rounded-lg bg-dark-800/50">
                  <span className="text-dark-300">{service.name}</span>
                  <Badge variant={service.status === 'online' ? 'success' : 'danger'}>
                    {service.status === 'online' ? 'Online' : 'Offline'}
                  </Badge>
                </div>
              ))}
            </div>
          </Card>

          {/* Save Button */}
          <div className="flex justify-end">
            <Button
              onClick={handleSave}
              isLoading={isSaving}
              icon={<Save className="w-5 h-5" />}
              size="lg"
            >
              Salva Impostazioni
            </Button>
          </div>
        </div>
      </motion.div>
    </div>
  );
}
