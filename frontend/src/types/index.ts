export interface VideoClip {
  id: number;
  path: string;
  url?: string;
  score: number;
  reason: string;
  duration: number;
  thumbnail?: string;
}

export interface ProcessingTask {
  task_id: string;
  status: 'PENDING' | 'PROGRESS' | 'SUCCESS' | 'FAILURE';
  progress?: {
    stage: string;
    progress: number;
  };
  result?: {
    status: string;
    clips: VideoClip[];
  };
  error?: string;
}

export interface UploadResponse {
  task_id: string;
  message: string;
}

export interface ApiSettings {
  llm_provider: 'openai' | 'openrouter' | 'anthropic' | 'google';
  llm_model: string;
  api_key: string;
  whisper_model: 'tiny' | 'base' | 'small' | 'medium' | 'large';
  use_local_whisper: boolean;
}

export interface VideoUploadState {
  file: File | null;
  preview: string | null;
  uploading: boolean;
  progress: number;
}
