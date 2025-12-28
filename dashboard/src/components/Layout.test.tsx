import { render, screen, fireEvent } from '@testing-library/react'
import { describe, it, expect } from 'vitest'
import { BrowserRouter } from 'react-router-dom'
import { Layout } from './Layout'

function renderWithRouter(ui: React.ReactElement) {
  return render(<BrowserRouter>{ui}</BrowserRouter>)
}

describe('Layout', () => {
  it('renders the app title', () => {
    renderWithRouter(<Layout />)

    expect(screen.getByText('SwingTrader')).toBeInTheDocument()
  })

  it('renders all navigation items', () => {
    renderWithRouter(<Layout />)

    expect(screen.getByText('Signals')).toBeInTheDocument()
    expect(screen.getByText('Portfolio')).toBeInTheDocument()
    expect(screen.getByText('Performance')).toBeInTheDocument()
    expect(screen.getByText('Risk')).toBeInTheDocument()
    expect(screen.getByText('Settings')).toBeInTheDocument()
  })

  it('renders navigation links with correct paths', () => {
    renderWithRouter(<Layout />)

    const links = screen.getAllByRole('link')
    const paths = links.map((link) => link.getAttribute('href'))

    expect(paths).toContain('/')
    expect(paths).toContain('/portfolio')
    expect(paths).toContain('/performance')
    expect(paths).toContain('/risk')
    expect(paths).toContain('/settings')
  })

  it('renders mobile menu button', () => {
    renderWithRouter(<Layout />)

    const menuButton = screen.getByRole('button', { name: /toggle menu/i })
    expect(menuButton).toBeInTheDocument()
  })

  it('toggles mobile menu on button click', () => {
    renderWithRouter(<Layout />)

    const menuButton = screen.getByRole('button', { name: /toggle menu/i })

    // Initially, mobile nav should be hidden (checking for duplicate "Signals" links)
    const initialSignalsLinks = screen.getAllByText('Signals')
    expect(initialSignalsLinks.length).toBe(1) // Desktop only

    // Click to open
    fireEvent.click(menuButton)

    // Now mobile nav should be visible
    const expandedSignalsLinks = screen.getAllByText('Signals')
    expect(expandedSignalsLinks.length).toBe(2) // Desktop + Mobile

    // Click to close
    fireEvent.click(menuButton)

    // Mobile nav should be hidden again
    const closedSignalsLinks = screen.getAllByText('Signals')
    expect(closedSignalsLinks.length).toBe(1)
  })

  it('closes mobile menu when link is clicked', () => {
    renderWithRouter(<Layout />)

    const menuButton = screen.getByRole('button', { name: /toggle menu/i })

    // Open mobile menu
    fireEvent.click(menuButton)

    // Click a mobile nav link (get the second occurrence which is in mobile menu)
    const mobileLinks = screen.getAllByText('Portfolio')
    fireEvent.click(mobileLinks[1])

    // Menu should close
    const signalsLinks = screen.getAllByText('Signals')
    expect(signalsLinks.length).toBe(1)
  })

  it('renders nav icons', () => {
    renderWithRouter(<Layout />)

    expect(screen.getByText('📊')).toBeInTheDocument()
    expect(screen.getByText('💼')).toBeInTheDocument()
    expect(screen.getByText('📈')).toBeInTheDocument()
    expect(screen.getByText('⚠️')).toBeInTheDocument()
    expect(screen.getByText('⚙️')).toBeInTheDocument()
  })
})
