import type {
  Signal,
  Trade,
  Portfolio,
  EquityPoint,
  PerformanceMetrics,
  RiskStatus,
  Settings,
} from '@/types'

const API_BASE = '/api'

class ApiError extends Error {
  status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = 'ApiError'
    this.status = status
  }
}

async function fetchApi<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const apiKey = localStorage.getItem('api_key') || ''
  const headers: HeadersInit = {
    'Content-Type': 'application/json',
    ...(apiKey && { 'X-API-Key': apiKey }),
    ...options.headers,
  }

  const response = await fetch(`${API_BASE}${endpoint}`, {
    ...options,
    headers,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Unknown error' }))
    throw new ApiError(response.status, error.detail || 'Request failed')
  }

  return response.json()
}

// Signals API
export async function getSignals(params?: {
  min_score?: number
  strategy?: string
  direction?: 'LONG' | 'SHORT'
}): Promise<Signal[]> {
  const searchParams = new URLSearchParams()
  if (params?.min_score) searchParams.set('min_score', params.min_score.toString())
  if (params?.strategy) searchParams.set('strategy', params.strategy)
  if (params?.direction) searchParams.set('direction', params.direction)

  const query = searchParams.toString()
  return fetchApi<Signal[]>(`/signals${query ? `?${query}` : ''}`)
}

export async function getSignal(id: string): Promise<Signal> {
  return fetchApi<Signal>(`/signals/${id}`)
}

// Trades API
export async function getTrades(params?: {
  start_date?: string
  end_date?: string
  status?: 'OPEN' | 'CLOSED' | 'STOPPED' | 'CANCELLED'
}): Promise<Trade[]> {
  const searchParams = new URLSearchParams()
  if (params?.start_date) searchParams.set('start_date', params.start_date)
  if (params?.end_date) searchParams.set('end_date', params.end_date)
  if (params?.status) searchParams.set('status', params.status)

  const query = searchParams.toString()
  return fetchApi<Trade[]>(`/trades${query ? `?${query}` : ''}`)
}

export async function getTrade(id: string): Promise<Trade> {
  return fetchApi<Trade>(`/trades/${id}`)
}

// Portfolio API
export async function getPortfolio(): Promise<Portfolio> {
  return fetchApi<Portfolio>('/portfolio')
}

export async function getEquityCurve(params?: {
  start_date?: string
  end_date?: string
}): Promise<EquityPoint[]> {
  const searchParams = new URLSearchParams()
  if (params?.start_date) searchParams.set('start_date', params.start_date)
  if (params?.end_date) searchParams.set('end_date', params.end_date)

  const query = searchParams.toString()
  return fetchApi<EquityPoint[]>(`/portfolio/equity-curve${query ? `?${query}` : ''}`)
}

// Performance API
export async function getPerformance(
  period: 'daily' | 'weekly' | 'monthly' | 'all' = 'all'
): Promise<PerformanceMetrics> {
  return fetchApi<PerformanceMetrics>(`/performance?period=${period}`)
}

// Risk API
export async function getRiskStatus(): Promise<RiskStatus> {
  return fetchApi<RiskStatus>('/risk')
}

export async function resetCircuitBreaker(name: string): Promise<{ message: string }> {
  return fetchApi<{ message: string }>(`/circuit-breakers/${name}/reset`, {
    method: 'POST',
  })
}

// Settings API
export async function getSettings(): Promise<Settings> {
  return fetchApi<Settings>('/settings')
}

export async function updateSettings(settings: Partial<Settings>): Promise<Settings> {
  return fetchApi<Settings>('/settings', {
    method: 'PUT',
    body: JSON.stringify(settings),
  })
}

// Health Check
export async function healthCheck(): Promise<{ status: string }> {
  return fetchApi<{ status: string }>('/health'.replace('/api', ''))
}

export { ApiError }
