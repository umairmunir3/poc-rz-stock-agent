import type { Signal } from '@/types'
import { format } from 'date-fns'

interface SignalCardProps {
  signal: Signal
  onClick?: () => void
}

export function SignalCard({ signal, onClick }: SignalCardProps) {
  const isLong = signal.direction === 'LONG'
  const statusColors = {
    PENDING: 'badge-info',
    TRIGGERED: 'badge-success',
    EXPIRED: 'badge-warning',
    CANCELLED: 'badge-danger',
  }

  return (
    <div
      className="card cursor-pointer hover:shadow-lg transition-shadow"
      onClick={onClick}
      role="button"
      tabIndex={0}
      onKeyDown={(e) => e.key === 'Enter' && onClick?.()}
    >
      <div className="flex justify-between items-start mb-3">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            {signal.symbol}
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400">{signal.strategy}</p>
        </div>
        <div className="flex flex-col items-end gap-1">
          <span
            className={`badge ${isLong ? 'badge-success' : 'badge-danger'}`}
          >
            {signal.direction}
          </span>
          <span className={`badge ${statusColors[signal.status]}`}>
            {signal.status}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4 mb-3">
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">Entry</p>
          <p className="font-medium text-gray-900 dark:text-white">
            ${signal.entry_price.toFixed(2)}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">Stop Loss</p>
          <p className="font-medium text-danger-600">${signal.stop_loss.toFixed(2)}</p>
        </div>
        <div>
          <p className="text-xs text-gray-500 dark:text-gray-400">Take Profit</p>
          <p className="font-medium text-primary-600">${signal.take_profit.toFixed(2)}</p>
        </div>
      </div>

      <div className="flex justify-between items-center">
        <div className="flex items-center">
          <span className="text-sm text-gray-500 dark:text-gray-400 mr-2">Score:</span>
          <div className="flex items-center">
            <div className="w-20 h-2 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
              <div
                className={`h-full ${
                  signal.score >= 80
                    ? 'bg-green-500'
                    : signal.score >= 60
                      ? 'bg-yellow-500'
                      : 'bg-red-500'
                }`}
                style={{ width: `${signal.score}%` }}
              />
            </div>
            <span className="ml-2 text-sm font-medium text-gray-900 dark:text-white">
              {signal.score}
            </span>
          </div>
        </div>
        <span className="text-xs text-gray-400">
          {format(new Date(signal.generated_at), 'MMM d, HH:mm')}
        </span>
      </div>
    </div>
  )
}
