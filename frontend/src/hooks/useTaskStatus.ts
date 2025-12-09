import { useState, useEffect, useCallback } from 'react';
import { ProcessingTask } from '../types';
import { getTaskStatus } from '../lib/api';

interface UseTaskStatusOptions {
  pollInterval?: number;
  enabled?: boolean;
}

export function useTaskStatus(
  taskId: string | null,
  options: UseTaskStatusOptions = {}
) {
  const { pollInterval = 2000, enabled = true } = options;
  const [task, setTask] = useState<ProcessingTask | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isPolling, setIsPolling] = useState(false);

  const fetchStatus = useCallback(async () => {
    if (!taskId) return;

    try {
      const status = await getTaskStatus(taskId);
      setTask(status);
      setError(null);

      // Stop polling if task is complete or failed
      if (status.status === 'SUCCESS' || status.status === 'FAILURE') {
        setIsPolling(false);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch status');
    }
  }, [taskId]);

  useEffect(() => {
    if (!taskId || !enabled) {
      setIsPolling(false);
      return;
    }

    setIsPolling(true);
    fetchStatus();

    const interval = setInterval(() => {
      if (isPolling) {
        fetchStatus();
      }
    }, pollInterval);

    return () => clearInterval(interval);
  }, [taskId, enabled, pollInterval, fetchStatus, isPolling]);

  const refetch = useCallback(() => {
    setIsPolling(true);
    fetchStatus();
  }, [fetchStatus]);

  return {
    task,
    error,
    isPolling,
    refetch,
  };
}
