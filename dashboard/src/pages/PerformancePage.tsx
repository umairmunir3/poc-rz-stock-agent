import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { getPerformance } from '@/api/client'
import type { PerformanceMetrics } from '@/types'

type Period = 'daily' | 'weekly' | 'monthly' | 'all'

interface MetricCardProps {
  label: string
  value: string | number
  suffix?: string
  positive?: boolean
  neutral?: boolean
}

function MetricCard({ label, value, suffix = '', positive, neutral }: MetricCardProps) {
  const colorClass = neutral
    ? 'text-gray-900 dark:text-white'
    : positive
      ? 'text-green-600'
      : 'text-red-600'

  return (
    <div className="card">
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">{label}</p>
      <p className={`text-2xl font-bold ${colorClass}`}>
        {value}
        {suffix && <span className="text-lg">{suffix}</span>}
      </p>
    </div>
  )
}

export function PerformancePage() {
  const [period, setPeriod] = useState<Period>('all')

  const { data: metrics, isLoading, error } = useQuery({
    queryKey: ['performance', period],
    queryFn: () => getPerformance(period),
  })

  const formatPercent = (value: number | undefined) => {
    if (value === undefined) return '-'
    return `${value >= 0 ? '+' : ''}${(value * 100).toFixed(2)}`
  }

  const formatRatio = (value: number | undefined) => {
    if (value === undefined) return '-'
    return value.toFixed(2)
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            Performance
          </h1>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Trading performance metrics
          </p>
        </div>

        {/* Period Selector */}
        <div className="flex rounded-lg overflow-hidden border border-gray-200 dark:border-gray-700">
          {(['daily', 'weekly', 'monthly', 'all'] as Period[]).map((p) => (
            <button
              key={p}
              onClick={() => setPeriod(p)}
              className={`px-4 py-2 text-sm font-medium transition-colors ${
                period === p
                  ? 'bg-primary-600 text-white'
                  : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700'
              }`}
            >
              {p.charAt(0).toUpperCase() + p.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {isLoading ? (
        <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
          {[...Array(10)].map((_, i) => (
            <div key={i} className="card animate-pulse">
              <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-2/3 mb-2" />
              <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-1/2" />
            </div>
          ))}
        </div>
      ) : error ? (
        <div className="card bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400">
          Failed to load performance metrics
        </div>
      ) : (
        <>
          {/* Return Metrics */}
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Returns
            </h2>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
              <MetricCard
                label="Total Return"
                value={formatPercent(metrics?.total_return)}
                suffix="%"
                positive={(metrics?.total_return || 0) >= 0}
              />
              <MetricCard
                label="CAGR"
                value={formatPercent(metrics?.cagr)}
                suffix="%"
                positive={(metrics?.cagr || 0) >= 0}
              />
              <MetricCard
                label="Max Drawdown"
                value={formatPercent(metrics?.max_drawdown)}
                suffix="%"
                positive={false}
              />
              <MetricCard
                label="Win Rate"
                value={formatPercent(metrics?.win_rate)}
                suffix="%"
                positive={(metrics?.win_rate || 0) >= 0.5}
              />
            </div>
          </div>

          {/* Risk-Adjusted Metrics */}
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Risk-Adjusted Returns
            </h2>
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
              <MetricCard
                label="Sharpe Ratio"
                value={formatRatio(metrics?.sharpe_ratio)}
                positive={(metrics?.sharpe_ratio || 0) >= 1}
                neutral={(metrics?.sharpe_ratio || 0) >= 0 && (metrics?.sharpe_ratio || 0) < 1}
              />
              <MetricCard
                label="Sortino Ratio"
                value={formatRatio(metrics?.sortino_ratio)}
                positive={(metrics?.sortino_ratio || 0) >= 1}
                neutral={(metrics?.sortino_ratio || 0) >= 0 && (metrics?.sortino_ratio || 0) < 1}
              />
              <MetricCard
                label="Profit Factor"
                value={formatRatio(metrics?.profit_factor)}
                positive={(metrics?.profit_factor || 0) >= 1.5}
                neutral={(metrics?.profit_factor || 0) >= 1 && (metrics?.profit_factor || 0) < 1.5}
              />
            </div>
          </div>

          {/* Trading Stats */}
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Trading Statistics
            </h2>
            <div className="grid grid-cols-2 lg:grid-cols-3 gap-4">
              <MetricCard
                label="Total Trades"
                value={metrics?.total_trades || 0}
                neutral
              />
              <MetricCard
                label="Avg Hold Days"
                value={formatRatio(metrics?.avg_hold_days)}
                neutral
              />
              <MetricCard
                label="Period"
                value={metrics?.period || period}
                neutral
              />
            </div>
          </div>

          {/* Performance Summary */}
          <div className="card">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Performance Summary
            </h2>
            <div className="prose dark:prose-invert max-w-none">
              <p className="text-gray-600 dark:text-gray-400">
                {getPerformanceSummary(metrics)}
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  )
}

function getPerformanceSummary(metrics: PerformanceMetrics | undefined): string {
  if (!metrics) return 'No performance data available.'

  const parts = []

  // Return assessment
  if (metrics.total_return >= 0.2) {
    parts.push('Excellent returns with strong positive performance.')
  } else if (metrics.total_return >= 0.1) {
    parts.push('Good returns with positive performance.')
  } else if (metrics.total_return >= 0) {
    parts.push('Modest positive returns.')
  } else {
    parts.push('Negative returns during this period.')
  }

  // Risk assessment
  if (metrics.sharpe_ratio >= 2) {
    parts.push('Outstanding risk-adjusted returns.')
  } else if (metrics.sharpe_ratio >= 1) {
    parts.push('Good risk-adjusted returns.')
  } else if (metrics.sharpe_ratio >= 0) {
    parts.push('Moderate risk-adjusted returns.')
  } else {
    parts.push('Poor risk-adjusted returns.')
  }

  // Win rate assessment
  if (metrics.win_rate >= 0.6) {
    parts.push('High win rate indicates consistent profitable trades.')
  } else if (metrics.win_rate >= 0.5) {
    parts.push('Win rate is acceptable but could be improved.')
  } else {
    parts.push('Win rate below 50% may require strategy adjustments.')
  }

  // Drawdown assessment
  if (Math.abs(metrics.max_drawdown) <= 0.1) {
    parts.push('Low drawdown shows good risk management.')
  } else if (Math.abs(metrics.max_drawdown) <= 0.2) {
    parts.push('Moderate drawdown is within acceptable limits.')
  } else {
    parts.push('High drawdown indicates elevated risk levels.')
  }

  return parts.join(' ')
}
