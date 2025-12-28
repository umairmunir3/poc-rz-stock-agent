import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect, vi } from 'vitest'
import { RiskMeter, CircuitBreakerStatus } from './RiskMeter'

describe('RiskMeter', () => {
  it('renders label and value', () => {
    render(<RiskMeter label="Portfolio Heat" value={50} />)

    expect(screen.getByText('Portfolio Heat')).toBeInTheDocument()
    expect(screen.getByText('50.0%')).toBeInTheDocument()
  })

  it('displays value without percent when showPercent is false', () => {
    render(<RiskMeter label="Daily P&L" value={2.5} showPercent={false} unit="%" />)

    expect(screen.getByText('2.5%')).toBeInTheDocument()
  })

  it('shows green color for low risk', () => {
    render(<RiskMeter label="Risk" value={30} max={100} />)

    const bar = document.querySelector('.bg-green-500')
    expect(bar).toBeInTheDocument()
  })

  it('shows yellow color for warning level', () => {
    render(
      <RiskMeter
        label="Risk"
        value={60}
        max={100}
        thresholds={{ warning: 50, danger: 75 }}
      />
    )

    const bar = document.querySelector('.bg-yellow-500')
    expect(bar).toBeInTheDocument()
  })

  it('shows red color for danger level', () => {
    render(
      <RiskMeter
        label="Risk"
        value={80}
        max={100}
        thresholds={{ warning: 50, danger: 75 }}
      />
    )

    const bar = document.querySelector('.bg-red-500')
    expect(bar).toBeInTheDocument()
  })

  it('caps percentage at 100', () => {
    render(<RiskMeter label="Risk" value={150} max={100} />)

    // Should cap at 100%
    expect(screen.getByText('100.0%')).toBeInTheDocument()
  })

  it('displays threshold markers', () => {
    render(
      <RiskMeter
        label="Risk"
        value={50}
        thresholds={{ warning: 50, danger: 75 }}
      />
    )

    // Check for threshold marker elements
    const markers = document.querySelectorAll('.absolute')
    expect(markers.length).toBeGreaterThan(0)
  })
})

describe('CircuitBreakerStatus', () => {
  it('renders breaker name and values', () => {
    render(
      <CircuitBreakerStatus
        name="Daily Loss Breaker"
        status="OK"
        threshold={5}
        currentValue={2}
      />
    )

    expect(screen.getByText('Daily Loss Breaker')).toBeInTheDocument()
    expect(screen.getByText('2.00 / 5.00')).toBeInTheDocument()
  })

  it('displays OK status badge', () => {
    render(
      <CircuitBreakerStatus
        name="Breaker"
        status="OK"
        threshold={5}
        currentValue={2}
      />
    )

    expect(screen.getByText('OK')).toBeInTheDocument()
  })

  it('displays TRIPPED status badge', () => {
    render(
      <CircuitBreakerStatus
        name="Breaker"
        status="TRIPPED"
        threshold={5}
        currentValue={6}
      />
    )

    expect(screen.getByText('TRIPPED')).toBeInTheDocument()
  })

  it('shows reset button when tripped', () => {
    const handleReset = vi.fn()
    render(
      <CircuitBreakerStatus
        name="Breaker"
        status="TRIPPED"
        threshold={5}
        currentValue={6}
        onReset={handleReset}
      />
    )

    const resetButton = screen.getByText('Reset')
    expect(resetButton).toBeInTheDocument()
  })

  it('does not show reset button when OK', () => {
    const handleReset = vi.fn()
    render(
      <CircuitBreakerStatus
        name="Breaker"
        status="OK"
        threshold={5}
        currentValue={2}
        onReset={handleReset}
      />
    )

    expect(screen.queryByText('Reset')).not.toBeInTheDocument()
  })

  it('calls onReset when reset button clicked', () => {
    const handleReset = vi.fn()
    render(
      <CircuitBreakerStatus
        name="Breaker"
        status="TRIPPED"
        threshold={5}
        currentValue={6}
        onReset={handleReset}
      />
    )

    fireEvent.click(screen.getByText('Reset'))
    expect(handleReset).toHaveBeenCalledTimes(1)
  })

  it('applies tripped styling', () => {
    render(
      <CircuitBreakerStatus
        name="Breaker"
        status="TRIPPED"
        threshold={5}
        currentValue={6}
      />
    )

    const card = document.querySelector('.border-red-500')
    expect(card).toBeInTheDocument()
  })

  it('applies OK styling', () => {
    render(
      <CircuitBreakerStatus
        name="Breaker"
        status="OK"
        threshold={5}
        currentValue={2}
      />
    )

    const card = document.querySelector('.border-green-500')
    expect(card).toBeInTheDocument()
  })
})
