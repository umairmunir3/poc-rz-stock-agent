import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import {
  getSignals,
  getSignal,
  getTrades,
  getPortfolio,
  getPerformance,
  getRiskStatus,
  getSettings,
  updateSettings,
  resetCircuitBreaker,
  ApiError,
} from './client'

// Mock fetch
const mockFetch = vi.fn()
// eslint-disable-next-line @typescript-eslint/no-explicit-any
;(globalThis as any).fetch = mockFetch

// Mock localStorage
const mockLocalStorage = {
  getItem: vi.fn(),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
}
Object.defineProperty(window, 'localStorage', { value: mockLocalStorage })

describe('API Client', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mockLocalStorage.getItem.mockReturnValue('test-api-key')
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  describe('getSignals', () => {
    it('fetches signals successfully', async () => {
      const mockSignals = [
        { id: 'SIG-001', symbol: 'AAPL' },
        { id: 'SIG-002', symbol: 'GOOGL' },
      ]
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockSignals),
      })

      const result = await getSignals()

      expect(mockFetch).toHaveBeenCalledWith('/api/signals', expect.any(Object))
      expect(result).toEqual(mockSignals)
    })

    it('includes API key in headers', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve([]),
      })

      await getSignals()

      expect(mockFetch).toHaveBeenCalledWith(
        '/api/signals',
        expect.objectContaining({
          headers: expect.objectContaining({
            'X-API-Key': 'test-api-key',
          }),
        })
      )
    })

    it('applies filters correctly', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve([]),
      })

      await getSignals({ min_score: 80, direction: 'LONG' })

      expect(mockFetch).toHaveBeenCalledWith(
        '/api/signals?min_score=80&direction=LONG',
        expect.any(Object)
      )
    })
  })

  describe('getSignal', () => {
    it('fetches single signal by ID', async () => {
      const mockSignal = { id: 'SIG-001', symbol: 'AAPL' }
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockSignal),
      })

      const result = await getSignal('SIG-001')

      expect(mockFetch).toHaveBeenCalledWith('/api/signals/SIG-001', expect.any(Object))
      expect(result).toEqual(mockSignal)
    })
  })

  describe('getTrades', () => {
    it('fetches trades with status filter', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve([]),
      })

      await getTrades({ status: 'OPEN' })

      expect(mockFetch).toHaveBeenCalledWith('/api/trades?status=OPEN', expect.any(Object))
    })

    it('fetches trades with date filters', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve([]),
      })

      await getTrades({ start_date: '2024-01-01', end_date: '2024-01-31' })

      expect(mockFetch).toHaveBeenCalledWith(
        '/api/trades?start_date=2024-01-01&end_date=2024-01-31',
        expect.any(Object)
      )
    })
  })

  describe('getPortfolio', () => {
    it('fetches portfolio data', async () => {
      const mockPortfolio = { total_value: 100000, cash: 50000 }
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockPortfolio),
      })

      const result = await getPortfolio()

      expect(mockFetch).toHaveBeenCalledWith('/api/portfolio', expect.any(Object))
      expect(result).toEqual(mockPortfolio)
    })
  })

  describe('getPerformance', () => {
    it('fetches performance metrics with period', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ period: 'monthly' }),
      })

      await getPerformance('monthly')

      expect(mockFetch).toHaveBeenCalledWith(
        '/api/performance?period=monthly',
        expect.any(Object)
      )
    })

    it('uses "all" period by default', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ period: 'all' }),
      })

      await getPerformance()

      expect(mockFetch).toHaveBeenCalledWith(
        '/api/performance?period=all',
        expect.any(Object)
      )
    })
  })

  describe('getRiskStatus', () => {
    it('fetches risk status', async () => {
      const mockRisk = { portfolio_heat: 45, circuit_breakers: [] }
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockRisk),
      })

      const result = await getRiskStatus()

      expect(mockFetch).toHaveBeenCalledWith('/api/risk', expect.any(Object))
      expect(result).toEqual(mockRisk)
    })
  })

  describe('getSettings', () => {
    it('fetches settings', async () => {
      const mockSettings = { risk_per_trade: 0.02, max_positions: 10 }
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockSettings),
      })

      const result = await getSettings()

      expect(mockFetch).toHaveBeenCalledWith('/api/settings', expect.any(Object))
      expect(result).toEqual(mockSettings)
    })
  })

  describe('updateSettings', () => {
    it('sends PUT request with settings', async () => {
      const newSettings = { max_positions: 15 }
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(newSettings),
      })

      await updateSettings(newSettings)

      expect(mockFetch).toHaveBeenCalledWith(
        '/api/settings',
        expect.objectContaining({
          method: 'PUT',
          body: JSON.stringify(newSettings),
        })
      )
    })
  })

  describe('resetCircuitBreaker', () => {
    it('sends POST request to reset breaker', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve({ message: 'Reset successful' }),
      })

      await resetCircuitBreaker('DailyLossBreaker')

      expect(mockFetch).toHaveBeenCalledWith(
        '/api/circuit-breakers/DailyLossBreaker/reset',
        expect.objectContaining({
          method: 'POST',
        })
      )
    })
  })

  describe('Error handling', () => {
    it('throws ApiError on non-ok response', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404,
        json: () => Promise.resolve({ detail: 'Not found' }),
      })

      await expect(getSignal('INVALID')).rejects.toThrow(ApiError)
    })

    it('includes status code in ApiError', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: () => Promise.resolve({ detail: 'Unauthorized' }),
      })

      try {
        await getSignals()
      } catch (error) {
        expect(error).toBeInstanceOf(ApiError)
        expect((error as ApiError).status).toBe(401)
      }
    })

    it('handles JSON parse errors gracefully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 500,
        json: () => Promise.reject(new Error('Invalid JSON')),
      })

      await expect(getSignals()).rejects.toThrow('Unknown error')
    })
  })
})
