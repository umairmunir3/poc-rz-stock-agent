import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { getSettings, updateSettings } from '@/api/client'
import type { Settings } from '@/types'

export function SettingsPage() {
  const queryClient = useQueryClient()
  const [formData, setFormData] = useState<Partial<Settings>>({})
  const [apiKey, setApiKey] = useState('')
  const [saved, setSaved] = useState(false)

  const { data: settings, isLoading, error } = useQuery({
    queryKey: ['settings'],
    queryFn: getSettings,
  })

  const updateMutation = useMutation({
    mutationFn: updateSettings,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] })
      setSaved(true)
      setTimeout(() => setSaved(false), 3000)
    },
  })

  useEffect(() => {
    if (settings) {
      setFormData(settings)
    }
  }, [settings])

  useEffect(() => {
    const storedKey = localStorage.getItem('api_key') || ''
    setApiKey(storedKey)
  }, [])

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    updateMutation.mutate(formData)
  }

  const handleApiKeyChange = (value: string) => {
    setApiKey(value)
    localStorage.setItem('api_key', value)
  }

  const handleStrategyToggle = (strategy: string) => {
    const current = formData.strategies_enabled || []
    const updated = current.includes(strategy)
      ? current.filter((s) => s !== strategy)
      : [...current, strategy]
    setFormData({ ...formData, strategies_enabled: updated })
  }

  const availableStrategies = [
    'RSI Mean Reversion',
    'Breakout',
    'Moving Average Crossover',
    'MACD Divergence',
    'Bollinger Band Squeeze',
  ]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Settings
        </h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Configure trading parameters and risk limits
        </p>
      </div>

      {isLoading ? (
        <div className="space-y-6">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="card animate-pulse">
              <div className="h-6 bg-gray-200 dark:bg-gray-700 rounded w-1/4 mb-4" />
              <div className="space-y-4">
                <div className="h-10 bg-gray-200 dark:bg-gray-700 rounded" />
                <div className="h-10 bg-gray-200 dark:bg-gray-700 rounded" />
              </div>
            </div>
          ))}
        </div>
      ) : error ? (
        <div className="card bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400">
          Failed to load settings
        </div>
      ) : (
        <form onSubmit={handleSubmit} className="space-y-6">
          {/* API Configuration */}
          <div className="card">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              API Configuration
            </h2>
            <div>
              <label className="label">API Key</label>
              <input
                type="password"
                value={apiKey}
                onChange={(e) => handleApiKeyChange(e.target.value)}
                className="input"
                placeholder="Enter your API key"
              />
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                Stored locally in your browser
              </p>
            </div>
          </div>

          {/* Position Sizing */}
          <div className="card">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Position Sizing
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="label">Risk Per Trade (%)</label>
                <input
                  type="number"
                  min={0.1}
                  max={10}
                  step={0.1}
                  value={(formData.risk_per_trade || 0) * 100}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      risk_per_trade: parseFloat(e.target.value) / 100,
                    })
                  }
                  className="input"
                />
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  Maximum: 10%
                </p>
              </div>
              <div>
                <label className="label">Max Positions</label>
                <input
                  type="number"
                  min={1}
                  max={50}
                  value={formData.max_positions || 10}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      max_positions: parseInt(e.target.value),
                    })
                  }
                  className="input"
                />
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  Maximum concurrent positions
                </p>
              </div>
              <div>
                <label className="label">Max Portfolio Risk (%)</label>
                <input
                  type="number"
                  min={1}
                  max={100}
                  step={1}
                  value={(formData.max_portfolio_risk || 0) * 100}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      max_portfolio_risk: parseFloat(e.target.value) / 100,
                    })
                  }
                  className="input"
                />
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                  Total portfolio risk limit
                </p>
              </div>
            </div>
          </div>

          {/* Circuit Breaker Thresholds */}
          <div className="card">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Circuit Breaker Thresholds
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <label className="label">Daily Loss Limit (%)</label>
                <input
                  type="number"
                  min={0.5}
                  max={10}
                  step={0.5}
                  value={(formData.circuit_breaker_thresholds?.daily_loss || 0) * 100}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      circuit_breaker_thresholds: {
                        ...formData.circuit_breaker_thresholds,
                        daily_loss: parseFloat(e.target.value) / 100,
                        weekly_loss: formData.circuit_breaker_thresholds?.weekly_loss || 0,
                        max_drawdown: formData.circuit_breaker_thresholds?.max_drawdown || 0,
                      },
                    })
                  }
                  className="input"
                />
              </div>
              <div>
                <label className="label">Weekly Loss Limit (%)</label>
                <input
                  type="number"
                  min={1}
                  max={20}
                  step={0.5}
                  value={(formData.circuit_breaker_thresholds?.weekly_loss || 0) * 100}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      circuit_breaker_thresholds: {
                        ...formData.circuit_breaker_thresholds,
                        daily_loss: formData.circuit_breaker_thresholds?.daily_loss || 0,
                        weekly_loss: parseFloat(e.target.value) / 100,
                        max_drawdown: formData.circuit_breaker_thresholds?.max_drawdown || 0,
                      },
                    })
                  }
                  className="input"
                />
              </div>
              <div>
                <label className="label">Max Drawdown (%)</label>
                <input
                  type="number"
                  min={5}
                  max={50}
                  step={1}
                  value={(formData.circuit_breaker_thresholds?.max_drawdown || 0) * 100}
                  onChange={(e) =>
                    setFormData({
                      ...formData,
                      circuit_breaker_thresholds: {
                        ...formData.circuit_breaker_thresholds,
                        daily_loss: formData.circuit_breaker_thresholds?.daily_loss || 0,
                        weekly_loss: formData.circuit_breaker_thresholds?.weekly_loss || 0,
                        max_drawdown: parseFloat(e.target.value) / 100,
                      },
                    })
                  }
                  className="input"
                />
              </div>
            </div>
          </div>

          {/* Strategy Selection */}
          <div className="card">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Enabled Strategies
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {availableStrategies.map((strategy) => (
                <label
                  key={strategy}
                  className="flex items-center p-3 border border-gray-200 dark:border-gray-700 rounded-lg cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800"
                >
                  <input
                    type="checkbox"
                    checked={formData.strategies_enabled?.includes(strategy) || false}
                    onChange={() => handleStrategyToggle(strategy)}
                    className="w-4 h-4 text-primary-600 border-gray-300 rounded focus:ring-primary-500"
                  />
                  <span className="ml-3 text-sm text-gray-900 dark:text-white">
                    {strategy}
                  </span>
                </label>
              ))}
            </div>
          </div>

          {/* Save Button */}
          <div className="flex items-center justify-end gap-4">
            {saved && (
              <span className="text-green-600 text-sm">Settings saved successfully!</span>
            )}
            {updateMutation.isError && (
              <span className="text-red-600 text-sm">Failed to save settings</span>
            )}
            <button
              type="submit"
              disabled={updateMutation.isPending}
              className="btn btn-primary"
            >
              {updateMutation.isPending ? 'Saving...' : 'Save Settings'}
            </button>
          </div>
        </form>
      )}
    </div>
  )
}
