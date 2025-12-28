// Signal Types
export interface Signal {
  id: string
  symbol: string
  strategy: string
  direction: 'LONG' | 'SHORT'
  entry_price: number
  stop_loss: number
  take_profit: number
  score: number
  reasoning: string
  timestamp: string
  metadata: Record<string, unknown>
}

// Trade Types
export interface Trade {
  id: string
  symbol: string
  direction: 'LONG' | 'SHORT'
  status: 'OPEN' | 'CLOSED'
  entry_price: number
  entry_date: string
  shares: number
  stop_loss: number
  take_profit: number
  exit_price: number | null
  exit_date: string | null
  pnl: number
  pnl_percent: number
}

// Portfolio Types
export interface Position {
  symbol: string
  shares: number
  entry_price: number
  current_price: number
  market_value: number
  unrealized_pnl: number
  unrealized_pnl_percent: number
}

export interface Portfolio {
  positions: Position[]
  total_value: number
  cash: number
  total_exposure: number
  exposure_percent: number
}

export interface EquityPoint {
  date: string
  equity: number
  drawdown: number
}

// Performance Types
export interface PerformanceMetrics {
  period: 'daily' | 'weekly' | 'monthly' | 'all'
  total_return: number
  cagr: number
  sharpe_ratio: number
  sortino_ratio: number
  max_drawdown: number
  win_rate: number
  profit_factor: number
  total_trades: number
  avg_hold_days: number
}

// Risk Types
export interface CircuitBreaker {
  name: string
  status: 'OK' | 'WARNING' | 'TRIGGERED'
  can_trade: boolean
  message: string
  triggered_at: string | null
}

export interface RiskStatus {
  circuit_breakers: CircuitBreaker[]
  portfolio_heat: number
  daily_pnl: number
  current_drawdown: number
}

// Settings Types
export interface Settings {
  risk_per_trade: number
  max_positions: number
  max_portfolio_risk: number
  strategies_enabled: string[]
  circuit_breaker_thresholds: {
    daily_loss: number
    weekly_loss: number
    max_drawdown: number
  }
}

// WebSocket Message Types
export interface WSMessage {
  type: 'signal' | 'price' | 'portfolio' | 'pong' | 'error'
  data?: unknown
  message?: string
}

export interface PriceUpdate {
  symbol: string
  price: number
  change: number
  change_percent: number
  volume: number
  timestamp: string
}
