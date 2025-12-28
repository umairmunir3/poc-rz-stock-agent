import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts'
import type { EquityPoint } from '@/types'
import { format } from 'date-fns'

interface EquityCurveProps {
  data: EquityPoint[]
  showDrawdown?: boolean
  height?: number
}

export function EquityCurve({ data, showDrawdown = true, height = 300 }: EquityCurveProps) {
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500 dark:text-gray-400">
        No data available
      </div>
    )
  }

  const formattedData = data.map((point) => ({
    ...point,
    date: format(new Date(point.date), 'MMM d'),
    drawdownPercent: point.drawdown * 100,
  }))

  const startingEquity = formattedData[0]?.equity || 0
  const currentEquity = formattedData[formattedData.length - 1]?.equity || 0
  const pnl = currentEquity - startingEquity
  const pnlPercent = startingEquity > 0 ? (pnl / startingEquity) * 100 : 0

  return (
    <div>
      <div className="flex justify-between items-center mb-4">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Equity Curve
          </h3>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Portfolio value over time
          </p>
        </div>
        <div className="text-right">
          <p className="text-2xl font-bold text-gray-900 dark:text-white">
            ${currentEquity.toLocaleString()}
          </p>
          <p
            className={`text-sm font-medium ${
              pnl >= 0 ? 'text-green-600' : 'text-red-600'
            }`}
          >
            {pnl >= 0 ? '+' : ''}${pnl.toLocaleString()} ({pnlPercent.toFixed(2)}%)
          </p>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={formattedData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="colorEquity" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#22c55e" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#22c55e" stopOpacity={0} />
            </linearGradient>
            {showDrawdown && (
              <linearGradient id="colorDrawdown" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
              </linearGradient>
            )}
          </defs>
          <CartesianGrid strokeDasharray="3 3" className="stroke-gray-200 dark:stroke-gray-700" />
          <XAxis
            dataKey="date"
            tick={{ fontSize: 12 }}
            className="text-gray-600 dark:text-gray-400"
          />
          <YAxis
            yAxisId="equity"
            orientation="left"
            tick={{ fontSize: 12 }}
            tickFormatter={(value: number) => `$${(value / 1000).toFixed(0)}k`}
            className="text-gray-600 dark:text-gray-400"
          />
          {showDrawdown && (
            <YAxis
              yAxisId="drawdown"
              orientation="right"
              tick={{ fontSize: 12 }}
              tickFormatter={(value: number) => `${value.toFixed(0)}%`}
              domain={[-20, 0]}
              className="text-gray-600 dark:text-gray-400"
            />
          )}
          <Tooltip
            contentStyle={{
              backgroundColor: 'rgba(255, 255, 255, 0.95)',
              border: '1px solid #e5e7eb',
              borderRadius: '8px',
            }}
            formatter={(value, name) => {
              if (value === undefined) return ['-', name || '']
              if (name === 'equity') {
                return [`$${Number(value).toLocaleString()}`, 'Equity']
              }
              return [`${Number(value).toFixed(2)}%`, 'Drawdown']
            }}
          />
          <ReferenceLine yAxisId="equity" y={startingEquity} stroke="#9ca3af" strokeDasharray="5 5" />
          <Area
            yAxisId="equity"
            type="monotone"
            dataKey="equity"
            stroke="#22c55e"
            fillOpacity={1}
            fill="url(#colorEquity)"
          />
          {showDrawdown && (
            <Area
              yAxisId="drawdown"
              type="monotone"
              dataKey="drawdownPercent"
              stroke="#ef4444"
              fillOpacity={1}
              fill="url(#colorDrawdown)"
            />
          )}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
