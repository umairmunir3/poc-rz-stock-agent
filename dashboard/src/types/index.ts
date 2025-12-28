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
  generated_at: string
  expires_at: string
  status: 'PENDING' | 'TRIGGERED' | 'EXPIRED' | 'CANCELLED'
  indicators: Record<string, number>
}

// Trade Types
export interface Trade {
  id: string
  signal_id: string
  symbol: string
  direction: 'LONG' | 'SHORT'
  entry_price: number
  exit_price: number | null
  quantity: number
  stop_loss: number
  take_profit: number
  status: 'OPEN' | 'CLOSED' | 'STOPPED' | 'CANCELLED'
  entry_time: string
  exit_time: string | null
  pnl: number | null
  pnl_percent: number | null
}

// Portfolio Types
export interface Position {
  symbol: string
  quantity: number
  entry_price: number
  current_price: number
  market_value: number
  unrealized_pnl: number
  unrealized_pnl_percent: number
  direction: 'LONG' | 'SHORT'
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
  status: 'OK' | 'TRIPPED'
  threshold: number
  current_value: number
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
