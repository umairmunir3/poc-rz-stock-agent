interface RiskMeterProps {
  value: number
  max?: number
  thresholds?: { warning: number; danger: number }
  label: string
  unit?: string
  showPercent?: boolean
}

export function RiskMeter({
  value,
  max = 100,
  thresholds = { warning: 50, danger: 75 },
  label,
  unit = '',
  showPercent = true,
}: RiskMeterProps) {
  const percentage = Math.min((value / max) * 100, 100)

  const getColor = () => {
    if (percentage >= thresholds.danger) return 'bg-red-500'
    if (percentage >= thresholds.warning) return 'bg-yellow-500'
    return 'bg-green-500'
  }

  const getTextColor = () => {
    if (percentage >= thresholds.danger) return 'text-red-600'
    if (percentage >= thresholds.warning) return 'text-yellow-600'
    return 'text-green-600'
  }

  return (
    <div className="card">
      <div className="flex justify-between items-center mb-2">
        <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
          {label}
        </span>
        <span className={`text-lg font-bold ${getTextColor()}`}>
          {showPercent ? `${percentage.toFixed(1)}%` : `${value}${unit}`}
        </span>
      </div>
      <div className="relative">
        <div className="w-full h-3 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
          <div
            className={`h-full transition-all duration-500 ${getColor()}`}
            style={{ width: `${percentage}%` }}
          />
        </div>
        {/* Threshold markers */}
        <div
          className="absolute top-0 h-3 w-0.5 bg-yellow-600"
          style={{ left: `${thresholds.warning}%` }}
        />
        <div
          className="absolute top-0 h-3 w-0.5 bg-red-600"
          style={{ left: `${thresholds.danger}%` }}
        />
      </div>
      <div className="flex justify-between text-xs text-gray-400 mt-1">
        <span>0</span>
        <span>{max}{unit}</span>
      </div>
    </div>
  )
}

interface CircuitBreakerStatusProps {
  name: string
  status: 'OK' | 'WARNING' | 'TRIGGERED'
  canTrade: boolean
  message: string
  triggeredAt?: string | null
  onReset?: () => void
}

export function CircuitBreakerStatus({
  name,
  status,
  canTrade,
  message,
  triggeredAt,
  onReset,
}: CircuitBreakerStatusProps) {
  const isTriggered = status === 'TRIGGERED'
  const isWarning = status === 'WARNING'

  const getBorderColor = () => {
    if (isTriggered) return 'border-red-500 bg-red-50 dark:bg-red-900/20'
    if (isWarning) return 'border-yellow-500 bg-yellow-50 dark:bg-yellow-900/20'
    return 'border-green-500'
  }

  const getBadgeClass = () => {
    if (isTriggered) return 'badge-danger'
    if (isWarning) return 'badge-warning'
    return 'badge-success'
  }

  return (
    <div className={`card border-l-4 ${getBorderColor()}`}>
      <div className="flex justify-between items-start">
        <div>
          <h4 className="font-medium text-gray-900 dark:text-white">{name}</h4>
          <p className="text-sm text-gray-500 dark:text-gray-400">{message}</p>
          {triggeredAt && (
            <p className="text-xs text-gray-400 mt-1">
              Triggered: {new Date(triggeredAt).toLocaleString()}
            </p>
          )}
        </div>
        <div className="flex items-center gap-2">
          <span className={`badge ${getBadgeClass()}`}>{status}</span>
          {isTriggered && onReset && (
            <button
              onClick={onReset}
              className="btn btn-danger text-xs py-1 px-2"
            >
              Reset
            </button>
          )}
        </div>
      </div>
      <div className="mt-2 flex items-center gap-2">
        <span
          className={`w-3 h-3 rounded-full ${canTrade ? 'bg-green-500' : 'bg-red-500'}`}
        />
        <span className="text-sm text-gray-600 dark:text-gray-400">
          {canTrade ? 'Trading allowed' : 'Trading blocked'}
        </span>
      </div>
    </div>
  )
}
