import { useState, useEffect, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import { SignalCard } from '@/components/SignalCard'
import { getSignals } from '@/api/client'
import { useWebSocket } from '@/hooks/useWebSocket'
import type { Signal, WSMessage } from '@/types'

export function SignalsPage() {
  const [filters, setFilters] = useState<{
    min_score?: number
    strategy?: string
    direction?: 'LONG' | 'SHORT'
  }>({})

  const [realtimeSignals, setRealtimeSignals] = useState<Signal[]>([])

  const { data: signals = [], isLoading, error, refetch } = useQuery({
    queryKey: ['signals', filters],
    queryFn: () => getSignals(filters),
  })

  const handleWebSocketMessage = useCallback((message: WSMessage) => {
    if (message.type === 'signal' && message.data) {
      setRealtimeSignals((prev) => {
        const signal = message.data as Signal
        const exists = prev.some((s) => s.id === signal.id)
        if (exists) {
          return prev.map((s) => (s.id === signal.id ? signal : s))
        }
        return [signal, ...prev].slice(0, 10)
      })
    }
  }, [])

  const { status: wsStatus } = useWebSocket('signals', {
    onMessage: handleWebSocketMessage,
  })

  // Merge realtime signals with fetched signals
  const allSignals = [...realtimeSignals, ...signals].reduce<Signal[]>((acc, signal) => {
    if (!acc.some((s) => s.id === signal.id)) {
      acc.push(signal)
    }
    return acc
  }, [])

  useEffect(() => {
    // Clear realtime signals when filters change
    setRealtimeSignals([])
  }, [filters])

  return (
    <div>
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center mb-6 gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Trading Signals
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Real-time trading opportunities
            <span
              className={`ml-2 inline-flex items-center ${
                wsStatus === 'connected' ? 'text-green-500' : 'text-gray-400'
              }`}
            >
              <span
                className={`w-2 h-2 rounded-full mr-1 ${
                  wsStatus === 'connected' ? 'bg-green-500 animate-pulse' : 'bg-gray-400'
                }`}
              />
              {wsStatus === 'connected' ? 'Live' : 'Offline'}
            </span>
          </p>
        </div>
        <button
          onClick={() => refetch()}
          className="btn btn-secondary"
        >
          Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="card mb-6">
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <div>
            <label className="label">Min Score</label>
            <input
              type="number"
              min={0}
              max={100}
              value={filters.min_score || ''}
              onChange={(e) =>
                setFilters((f) => ({
                  ...f,
                  min_score: e.target.value ? parseInt(e.target.value) : undefined,
                }))
              }
              className="input"
              placeholder="0-100"
            />
          </div>
          <div>
            <label className="label">Strategy</label>
            <select
              value={filters.strategy || ''}
              onChange={(e) =>
                setFilters((f) => ({
                  ...f,
                  strategy: e.target.value || undefined,
                }))
              }
              className="input"
            >
              <option value="">All Strategies</option>
              <option value="RSI Mean Reversion">RSI Mean Reversion</option>
              <option value="Breakout">Breakout</option>
              <option value="Moving Average Crossover">Moving Average Crossover</option>
            </select>
          </div>
          <div>
            <label className="label">Direction</label>
            <select
              value={filters.direction || ''}
              onChange={(e) =>
                setFilters((f) => ({
                  ...f,
                  direction: (e.target.value as 'LONG' | 'SHORT') || undefined,
                }))
              }
              className="input"
            >
              <option value="">All Directions</option>
              <option value="LONG">Long</option>
              <option value="SHORT">Short</option>
            </select>
          </div>
        </div>
      </div>

      {/* Signals List */}
      {isLoading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="card animate-pulse">
              <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-1/3 mb-3" />
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-4" />
              <div className="grid grid-cols-3 gap-4 mb-3">
                <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded" />
                <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded" />
                <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded" />
              </div>
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-full" />
            </div>
          ))}
        </div>
      ) : error ? (
        <div className="card bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400">
          Failed to load signals. Please try again.
        </div>
      ) : allSignals.length === 0 ? (
        <div className="card text-center text-gray-500 dark:text-gray-400 py-12">
          No signals found matching your criteria
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {allSignals.map((signal) => (
            <SignalCard
              key={signal.id}
              signal={signal}
              onClick={() => console.log('Signal clicked:', signal.id)}
            />
          ))}
        </div>
      )}
    </div>
  )
}
