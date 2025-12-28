import { render, screen } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { EquityCurve } from './EquityCurve'
import type { EquityPoint } from '@/types'

const mockData: EquityPoint[] = [
  { date: '2024-01-01', equity: 10000, drawdown: 0 },
  { date: '2024-01-02', equity: 10500, drawdown: 0 },
  { date: '2024-01-03', equity: 10200, drawdown: -0.0286 },
  { date: '2024-01-04', equity: 10800, drawdown: 0 },
  { date: '2024-01-05', equity: 11000, drawdown: 0 },
]

describe('EquityCurve', () => {
  it('renders title and description', () => {
    render(<EquityCurve data={mockData} />)

    expect(screen.getByText('Equity Curve')).toBeInTheDocument()
    expect(screen.getByText('Portfolio value over time')).toBeInTheDocument()
  })

  it('displays current equity value', () => {
    render(<EquityCurve data={mockData} />)

    // Last equity is 11000
    expect(screen.getByText('$11,000')).toBeInTheDocument()
  })

  it('displays P&L information', () => {
    render(<EquityCurve data={mockData} />)

    // P&L is 11000 - 10000 = 1000 (10%)
    expect(screen.getByText('+$1,000 (10.00%)')).toBeInTheDocument()
  })

  it('shows empty state when no data', () => {
    render(<EquityCurve data={[]} />)

    expect(screen.getByText('No data available')).toBeInTheDocument()
  })

  it('renders chart container', () => {
    render(<EquityCurve data={mockData} />)

    // Recharts ResponsiveContainer should be present
    const chart = document.querySelector('.recharts-responsive-container')
    expect(chart).toBeInTheDocument()
  })

  it('uses custom height when provided', () => {
    render(<EquityCurve data={mockData} height={400} />)

    const chart = document.querySelector('.recharts-responsive-container')
    expect(chart).toBeInTheDocument()
  })

  it('hides drawdown when showDrawdown is false', () => {
    render(<EquityCurve data={mockData} showDrawdown={false} />)

    // Should still render the chart
    expect(screen.getByText('Equity Curve')).toBeInTheDocument()
  })

  it('shows negative P&L correctly', () => {
    const negativeData: EquityPoint[] = [
      { date: '2024-01-01', equity: 10000, drawdown: 0 },
      { date: '2024-01-02', equity: 9000, drawdown: -0.1 },
    ]
    render(<EquityCurve data={negativeData} />)

    // P&L is -1000 (-10%) - text may be split across elements
    expect(screen.getByText(/\$9,000/)).toBeInTheDocument()
    expect(screen.getByText(/-10\.00/)).toBeInTheDocument()
  })

  it('handles single data point', () => {
    const singlePoint: EquityPoint[] = [
      { date: '2024-01-01', equity: 10000, drawdown: 0 },
    ]
    render(<EquityCurve data={singlePoint} />)

    expect(screen.getByText('$10,000')).toBeInTheDocument()
    expect(screen.getByText('+$0 (0.00%)')).toBeInTheDocument()
  })
})
