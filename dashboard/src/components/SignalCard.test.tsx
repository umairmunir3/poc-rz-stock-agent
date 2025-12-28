import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { SignalCard } from './SignalCard'
import type { Signal } from '@/types'

const mockSignal: Signal = {
  id: 'SIG-001',
  symbol: 'AAPL',
  strategy: 'RSI Mean Reversion',
  direction: 'LONG',
  entry_price: 150.0,
  stop_loss: 145.0,
  take_profit: 160.0,
  score: 85,
  generated_at: '2024-01-15T10:30:00Z',
  expires_at: '2024-01-16T10:30:00Z',
  status: 'PENDING',
  indicators: { rsi: 30, macd: 0.5 },
}

describe('SignalCard', () => {
  it('renders signal symbol and strategy', () => {
    render(<SignalCard signal={mockSignal} />)

    expect(screen.getByText('AAPL')).toBeInTheDocument()
    expect(screen.getByText('RSI Mean Reversion')).toBeInTheDocument()
  })

  it('displays direction badge correctly', () => {
    render(<SignalCard signal={mockSignal} />)

    expect(screen.getByText('LONG')).toBeInTheDocument()
  })

  it('displays SHORT direction for short signals', () => {
    const shortSignal: Signal = { ...mockSignal, direction: 'SHORT' }
    render(<SignalCard signal={shortSignal} />)

    expect(screen.getByText('SHORT')).toBeInTheDocument()
  })

  it('displays price levels correctly', () => {
    render(<SignalCard signal={mockSignal} />)

    expect(screen.getByText('$150.00')).toBeInTheDocument()
    expect(screen.getByText('$145.00')).toBeInTheDocument()
    expect(screen.getByText('$160.00')).toBeInTheDocument()
  })

  it('displays score correctly', () => {
    render(<SignalCard signal={mockSignal} />)

    expect(screen.getByText('85')).toBeInTheDocument()
  })

  it('displays status badge', () => {
    render(<SignalCard signal={mockSignal} />)

    expect(screen.getByText('PENDING')).toBeInTheDocument()
  })

  it('calls onClick when clicked', () => {
    const handleClick = vi.fn()
    render(<SignalCard signal={mockSignal} onClick={handleClick} />)

    const card = screen.getByRole('button')
    fireEvent.click(card)

    expect(handleClick).toHaveBeenCalledTimes(1)
  })

  it('calls onClick on Enter key press', () => {
    const handleClick = vi.fn()
    render(<SignalCard signal={mockSignal} onClick={handleClick} />)

    const card = screen.getByRole('button')
    fireEvent.keyDown(card, { key: 'Enter' })

    expect(handleClick).toHaveBeenCalledTimes(1)
  })

  it('renders different status colors', () => {
    const triggeredSignal: Signal = { ...mockSignal, status: 'TRIGGERED' }
    render(<SignalCard signal={triggeredSignal} />)

    expect(screen.getByText('TRIGGERED')).toBeInTheDocument()
  })

  it('shows high score with green indicator', () => {
    render(<SignalCard signal={mockSignal} />)

    // Score is 85, which should be >= 80 and show green
    const scoreBar = document.querySelector('.bg-green-500')
    expect(scoreBar).toBeInTheDocument()
  })

  it('shows medium score with yellow indicator', () => {
    const mediumScoreSignal: Signal = { ...mockSignal, score: 65 }
    render(<SignalCard signal={mediumScoreSignal} />)

    const scoreBar = document.querySelector('.bg-yellow-500')
    expect(scoreBar).toBeInTheDocument()
  })

  it('shows low score with red indicator', () => {
    const lowScoreSignal: Signal = { ...mockSignal, score: 40 }
    render(<SignalCard signal={lowScoreSignal} />)

    const scoreBar = document.querySelector('.bg-red-500')
    expect(scoreBar).toBeInTheDocument()
  })
})
