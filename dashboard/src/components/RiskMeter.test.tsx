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
  it('renders breaker name and message', () => {
    render(
      <CircuitBreakerStatus
        name="Daily Loss Breaker"
        status="OK"
        canTrade={true}
        message="Within acceptable limits"
      />
    )

    expect(screen.getByText('Daily Loss Breaker')).toBeInTheDocument()
    expect(screen.getByText('Within acceptable limits')).toBeInTheDocument()
  })

  it('displays OK status badge', () => {
    render(
      <CircuitBreakerStatus
        name="Breaker"
        status="OK"
        canTrade={true}
        message="All good"
      />
    )

    expect(screen.getByText('OK')).toBeInTheDocument()
  })

  it('displays TRIGGERED status badge', () => {
    render(
      <CircuitBreakerStatus
        name="Breaker"
        status="TRIGGERED"
        canTrade={false}
        message="Daily loss limit exceeded"
      />
    )

    expect(screen.getByText('TRIGGERED')).toBeInTheDocument()
  })

  it('displays WARNING status badge', () => {
    render(
      <CircuitBreakerStatus
        name="Breaker"
        status="WARNING"
        canTrade={true}
        message="Approaching limit"
      />
    )

    expect(screen.getByText('WARNING')).toBeInTheDocument()
  })

  it('shows reset button when triggered', () => {
    const handleReset = vi.fn()
    render(
      <CircuitBreakerStatus
        name="Breaker"
        status="TRIGGERED"
        canTrade={false}
        message="Limit exceeded"
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
        canTrade={true}
        message="All good"
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
        status="TRIGGERED"
        canTrade={false}
        message="Limit exceeded"
        onReset={handleReset}
      />
    )

    fireEvent.click(screen.getByText('Reset'))
    expect(handleReset).toHaveBeenCalledTimes(1)
  })

  it('applies triggered styling', () => {
    render(
      <CircuitBreakerStatus
        name="Breaker"
        status="TRIGGERED"
        canTrade={false}
        message="Limit exceeded"
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
        canTrade={true}
        message="All good"
      />
    )

    const card = document.querySelector('.border-green-500')
    expect(card).toBeInTheDocument()
  })

  it('applies warning styling', () => {
    render(
      <CircuitBreakerStatus
        name="Breaker"
        status="WARNING"
        canTrade={true}
        message="Approaching limit"
      />
    )

    const card = document.querySelector('.border-yellow-500')
    expect(card).toBeInTheDocument()
  })

  it('shows trading allowed indicator when canTrade is true', () => {
    render(
      <CircuitBreakerStatus
        name="Breaker"
        status="OK"
        canTrade={true}
        message="All good"
      />
    )

    expect(screen.getByText('Trading allowed')).toBeInTheDocument()
  })

  it('shows trading blocked indicator when canTrade is false', () => {
    render(
      <CircuitBreakerStatus
        name="Breaker"
        status="TRIGGERED"
        canTrade={false}
        message="Limit exceeded"
      />
    )

    expect(screen.getByText('Trading blocked')).toBeInTheDocument()
  })

  it('displays triggered timestamp when provided', () => {
    render(
      <CircuitBreakerStatus
        name="Breaker"
        status="TRIGGERED"
        canTrade={false}
        message="Limit exceeded"
        triggeredAt="2024-01-15T10:30:00Z"
      />
    )

    expect(screen.getByText(/Triggered:/)).toBeInTheDocument()
  })
})
