import { useQuery } from '@tanstack/react-query'
import { EquityCurve } from '@/components/EquityCurve'
import { getPortfolio, getEquityCurve, getTrades } from '@/api/client'

export function PortfolioPage() {
  const { data: portfolio, isLoading: portfolioLoading } = useQuery({
    queryKey: ['portfolio'],
    queryFn: getPortfolio,
  })

  const { data: equityData = [], isLoading: equityLoading } = useQuery({
    queryKey: ['equity-curve'],
    queryFn: () => getEquityCurve(),
  })

  const { data: trades = [], isLoading: tradesLoading } = useQuery({
    queryKey: ['trades', { status: 'OPEN' }],
    queryFn: () => getTrades({ status: 'OPEN' }),
  })

  const isLoading = portfolioLoading || equityLoading || tradesLoading

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
          Portfolio
        </h1>
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Current positions and performance
        </p>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="card">
          <p className="text-sm text-gray-500 dark:text-gray-400">Total Value</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            {portfolioLoading ? (
              <span className="animate-pulse">...</span>
            ) : (
              `$${portfolio?.total_value.toLocaleString() || 0}`
            )}
          </p>
        </div>
        <div className="card">
          <p className="text-sm text-gray-500 dark:text-gray-400">Cash</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            {portfolioLoading ? (
              <span className="animate-pulse">...</span>
            ) : (
              `$${portfolio?.cash.toLocaleString() || 0}`
            )}
          </p>
        </div>
        <div className="card">
          <p className="text-sm text-gray-500 dark:text-gray-400">Exposure</p>
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            {portfolioLoading ? (
              <span className="animate-pulse">...</span>
            ) : (
              `$${portfolio?.total_exposure.toLocaleString() || 0}`
            )}
          </p>
        </div>
        <div className="card">
          <p className="text-sm text-gray-500 dark:text-gray-400">Exposure %</p>
          <p
            className={`text-2xl font-bold ${
              (portfolio?.exposure_percent || 0) > 80
                ? 'text-red-600'
                : (portfolio?.exposure_percent || 0) > 50
                  ? 'text-yellow-600'
                  : 'text-green-600'
            }`}
          >
            {portfolioLoading ? (
              <span className="animate-pulse">...</span>
            ) : (
              `${portfolio?.exposure_percent.toFixed(1) || 0}%`
            )}
          </p>
        </div>
      </div>

      {/* Equity Curve */}
      <div className="card">
        {equityLoading ? (
          <div className="h-64 animate-pulse bg-gray-200 dark:bg-gray-700 rounded" />
        ) : (
          <EquityCurve data={equityData} />
        )}
      </div>

      {/* Open Positions */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Open Positions
        </h2>
        {isLoading ? (
          <div className="space-y-3">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-12 animate-pulse bg-gray-200 dark:bg-gray-700 rounded" />
            ))}
          </div>
        ) : portfolio?.positions.length === 0 ? (
          <p className="text-gray-500 dark:text-gray-400 text-center py-8">
            No open positions
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="text-left text-sm text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                  <th className="pb-2">Symbol</th>
                  <th className="pb-2">Direction</th>
                  <th className="pb-2">Qty</th>
                  <th className="pb-2">Entry</th>
                  <th className="pb-2">Current</th>
                  <th className="pb-2">Value</th>
                  <th className="pb-2 text-right">P&L</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {portfolio?.positions.map((position) => (
                  <tr key={position.symbol} className="text-sm">
                    <td className="py-3 font-medium text-gray-900 dark:text-white">
                      {position.symbol}
                    </td>
                    <td className="py-3">
                      <span
                        className={`badge ${
                          position.direction === 'LONG' ? 'badge-success' : 'badge-danger'
                        }`}
                      >
                        {position.direction}
                      </span>
                    </td>
                    <td className="py-3 text-gray-700 dark:text-gray-300">
                      {position.quantity}
                    </td>
                    <td className="py-3 text-gray-700 dark:text-gray-300">
                      ${position.entry_price.toFixed(2)}
                    </td>
                    <td className="py-3 text-gray-700 dark:text-gray-300">
                      ${position.current_price.toFixed(2)}
                    </td>
                    <td className="py-3 text-gray-700 dark:text-gray-300">
                      ${position.market_value.toLocaleString()}
                    </td>
                    <td
                      className={`py-3 text-right font-medium ${
                        position.unrealized_pnl >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}
                    >
                      {position.unrealized_pnl >= 0 ? '+' : ''}${position.unrealized_pnl.toFixed(2)}
                      <span className="text-xs ml-1">
                        ({position.unrealized_pnl_percent.toFixed(2)}%)
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Recent Trades */}
      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Open Trades
        </h2>
        {tradesLoading ? (
          <div className="space-y-3">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-12 animate-pulse bg-gray-200 dark:bg-gray-700 rounded" />
            ))}
          </div>
        ) : trades.length === 0 ? (
          <p className="text-gray-500 dark:text-gray-400 text-center py-8">
            No open trades
          </p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full">
              <thead>
                <tr className="text-left text-sm text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                  <th className="pb-2">Symbol</th>
                  <th className="pb-2">Direction</th>
                  <th className="pb-2">Entry</th>
                  <th className="pb-2">Stop Loss</th>
                  <th className="pb-2">Take Profit</th>
                  <th className="pb-2">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                {trades.map((trade) => (
                  <tr key={trade.id} className="text-sm">
                    <td className="py-3 font-medium text-gray-900 dark:text-white">
                      {trade.symbol}
                    </td>
                    <td className="py-3">
                      <span
                        className={`badge ${
                          trade.direction === 'LONG' ? 'badge-success' : 'badge-danger'
                        }`}
                      >
                        {trade.direction}
                      </span>
                    </td>
                    <td className="py-3 text-gray-700 dark:text-gray-300">
                      ${trade.entry_price.toFixed(2)}
                    </td>
                    <td className="py-3 text-red-600">${trade.stop_loss.toFixed(2)}</td>
                    <td className="py-3 text-green-600">${trade.take_profit.toFixed(2)}</td>
                    <td className="py-3">
                      <span className="badge badge-info">{trade.status}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  )
}
