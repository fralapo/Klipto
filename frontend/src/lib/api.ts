import axios from 'axios';
import { ProcessingTask, UploadResponse, ApiSettings } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const uploadVideo = async (
  file: File,
  numClips: number = 3,
  onProgress?: (progress: number) => void
): Promise<UploadResponse> => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await api.post<UploadResponse>(
    `/upload?num_clips=${numClips}`,
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent) => {
        if (progressEvent.total && onProgress) {
          const progress = Math.round((progressEvent.loaded * 100) / progressEvent.total);
          onProgress(progress);
        }
      },
    }
  );

  return response.data;
};

export const getTaskStatus = async (taskId: string): Promise<ProcessingTask> => {
  const response = await api.get<ProcessingTask>(`/status/${taskId}`);
  return response.data;
};

export const getClipUrl = (clipPath: string): string => {
  // Convert absolute path to API endpoint
  const filename = clipPath.split('/').pop();
  return `${API_BASE_URL}/clips/${filename}`;
};

export const getSettings = async (): Promise<ApiSettings> => {
  const response = await api.get<ApiSettings>('/settings');
  return response.data;
};

export const updateSettings = async (settings: Partial<ApiSettings>): Promise<ApiSettings> => {
  const response = await api.post<ApiSettings>('/settings', settings);
  return response.data;
};

export const getTasks = async (): Promise<ProcessingTask[]> => {
  const response = await api.get<ProcessingTask[]>('/tasks');
  return response.data;
};

export const deleteClip = async (clipId: string): Promise<void> => {
  await api.delete(`/clips/${clipId}`);
};

export const downloadClip = (clipPath: string): void => {
  const url = getClipUrl(clipPath);
  const link = document.createElement('a');
  link.href = url;
  link.download = clipPath.split('/').pop() || 'clip.mp4';
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
};

export default api;
