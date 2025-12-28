import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { RiskMeter, CircuitBreakerStatus } from '@/components/RiskMeter'
import { getRiskStatus, resetCircuitBreaker } from '@/api/client'

export function RiskPage() {
  const queryClient = useQueryClient()

  const { data: riskStatus, isLoading, error } = useQuery({
    queryKey: ['risk'],
    queryFn: getRiskStatus,
    refetchInterval: 30000, // Refresh every 30 seconds
  })

  const resetMutation = useMutation({
    mutationFn: resetCircuitBreaker,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['risk'] })
    },
  })

  const handleResetBreaker = (name: string) => {
    if (confirm(`Are you sure you want to reset the ${name}?`)) {
      resetMutation.mutate(name)
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Risk Management
        </h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Monitor risk levels and circuit breakers
        </p>
      </div>

      {isLoading ? (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="card animate-pulse">
                <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/2 mb-2" />
                <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-1/3 mb-4" />
                <div className="h-3 bg-gray-200 dark:bg-gray-700 rounded w-full" />
              </div>
            ))}
          </div>
        </div>
      ) : error ? (
        <div className="card bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400">
          Failed to load risk status
        </div>
      ) : (
        <>
          {/* Risk Meters */}
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Risk Indicators
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <RiskMeter
                label="Portfolio Heat"
                value={riskStatus?.portfolio_heat || 0}
                max={100}
                thresholds={{ warning: 50, danger: 75 }}
                showPercent
              />
              <RiskMeter
                label="Daily P&L"
                value={Math.abs(riskStatus?.daily_pnl || 0)}
                max={5}
                thresholds={{ warning: 2, danger: 4 }}
                unit="%"
                showPercent={false}
              />
              <RiskMeter
                label="Current Drawdown"
                value={Math.abs(riskStatus?.current_drawdown || 0) * 100}
                max={20}
                thresholds={{ warning: 10, danger: 15 }}
                showPercent
              />
            </div>
          </div>

          {/* Summary Cards */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="card">
              <p className="text-sm text-gray-500 dark:text-gray-400">Portfolio Heat</p>
              <p
                className={`text-2xl font-bold ${
                  (riskStatus?.portfolio_heat || 0) > 75
                    ? 'text-red-600'
                    : (riskStatus?.portfolio_heat || 0) > 50
                      ? 'text-yellow-600'
                      : 'text-green-600'
                }`}
              >
                {riskStatus?.portfolio_heat.toFixed(1)}%
              </p>
            </div>
            <div className="card">
              <p className="text-sm text-gray-500 dark:text-gray-400">Daily P&L</p>
              <p
                className={`text-2xl font-bold ${
                  (riskStatus?.daily_pnl || 0) >= 0 ? 'text-green-600' : 'text-red-600'
                }`}
              >
                {(riskStatus?.daily_pnl || 0) >= 0 ? '+' : ''}
                {riskStatus?.daily_pnl.toFixed(2)}%
              </p>
            </div>
            <div className="card">
              <p className="text-sm text-gray-500 dark:text-gray-400">Current Drawdown</p>
              <p className="text-2xl font-bold text-red-600">
                {((riskStatus?.current_drawdown || 0) * 100).toFixed(2)}%
              </p>
            </div>
            <div className="card">
              <p className="text-sm text-gray-500 dark:text-gray-400">Active Breakers</p>
              <p
                className={`text-2xl font-bold ${
                  riskStatus?.circuit_breakers.some((cb) => cb.status === 'TRIGGERED')
                    ? 'text-red-600'
                    : 'text-green-600'
                }`}
              >
                {riskStatus?.circuit_breakers.filter((cb) => cb.status === 'TRIGGERED').length || 0}
              </p>
            </div>
          </div>

          {/* Circuit Breakers */}
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Circuit Breakers
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {riskStatus?.circuit_breakers.map((breaker) => (
                <CircuitBreakerStatus
                  key={breaker.name}
                  name={breaker.name}
                  status={breaker.status}
                  canTrade={breaker.can_trade}
                  message={breaker.message}
                  triggeredAt={breaker.triggered_at}
                  onReset={() => handleResetBreaker(breaker.name)}
                />
              ))}
            </div>
          </div>

          {/* Risk Guidelines */}
          <div className="card">
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Risk Guidelines
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h3 className="font-medium text-gray-900 dark:text-white mb-2">
                  Portfolio Heat Levels
                </h3>
                <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                  <li className="flex items-center">
                    <span className="w-3 h-3 rounded-full bg-green-500 mr-2" />
                    0-50%: Normal operating range
                  </li>
                  <li className="flex items-center">
                    <span className="w-3 h-3 rounded-full bg-yellow-500 mr-2" />
                    50-75%: Elevated risk, reduce position sizes
                  </li>
                  <li className="flex items-center">
                    <span className="w-3 h-3 rounded-full bg-red-500 mr-2" />
                    75-100%: High risk, avoid new positions
                  </li>
                </ul>
              </div>
              <div>
                <h3 className="font-medium text-gray-900 dark:text-white mb-2">
                  Circuit Breaker Actions
                </h3>
                <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                  <li>• Daily Loss Breaker: Halts new trades for the day</li>
                  <li>• Weekly Loss Breaker: Reduces position sizes by 50%</li>
                  <li>• Drawdown Breaker: Closes all positions</li>
                  <li>• Volatility Breaker: Widens stop losses</li>
                </ul>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  )
}
