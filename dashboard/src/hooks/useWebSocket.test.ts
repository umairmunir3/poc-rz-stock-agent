import { renderHook, act, waitFor } from '@testing-library/react'
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { useWebSocket } from './useWebSocket'

// Mock WebSocket
class MockWebSocket {
  static instances: MockWebSocket[] = []

  url: string
  readyState: number = WebSocket.CONNECTING
  onopen: (() => void) | null = null
  onclose: (() => void) | null = null
  onmessage: ((event: { data: string }) => void) | null = null
  onerror: ((event: Event) => void) | null = null

  constructor(url: string) {
    this.url = url
    MockWebSocket.instances.push(this)
  }

  send = vi.fn()
  close = vi.fn(() => {
    this.readyState = WebSocket.CLOSED
  })

  // Helpers for tests
  simulateOpen() {
    this.readyState = WebSocket.OPEN
    this.onopen?.()
  }

  simulateMessage(data: object) {
    this.onmessage?.({ data: JSON.stringify(data) })
  }

  simulateClose() {
    this.readyState = WebSocket.CLOSED
    this.onclose?.()
  }

  simulateError() {
    this.onerror?.(new Event('error'))
  }
}

// Mock localStorage
const mockLocalStorage = {
  getItem: vi.fn().mockReturnValue('test-token'),
  setItem: vi.fn(),
  removeItem: vi.fn(),
  clear: vi.fn(),
}
Object.defineProperty(window, 'localStorage', { value: mockLocalStorage })

describe('useWebSocket', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    MockWebSocket.instances = []
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    ;(globalThis as any).WebSocket = MockWebSocket
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('connects to WebSocket on mount', () => {
    renderHook(() => useWebSocket('signals'))

    expect(MockWebSocket.instances.length).toBe(1)
    expect(MockWebSocket.instances[0].url).toContain('/ws/signals')
  })

  it('includes token in connection URL', () => {
    renderHook(() => useWebSocket('signals'))

    expect(MockWebSocket.instances[0].url).toContain('token=test-token')
  })

  it('starts with connecting status', () => {
    const { result } = renderHook(() => useWebSocket('signals'))

    expect(result.current.status).toBe('connecting')
  })

  it('updates status to connected on open', async () => {
    const { result } = renderHook(() => useWebSocket('signals'))

    act(() => {
      MockWebSocket.instances[0].simulateOpen()
    })

    await waitFor(() => {
      expect(result.current.status).toBe('connected')
    })
  })

  it('updates status to disconnected on close', async () => {
    const { result } = renderHook(() => useWebSocket('signals'))

    act(() => {
      MockWebSocket.instances[0].simulateOpen()
    })

    await waitFor(() => {
      expect(result.current.status).toBe('connected')
    })

    act(() => {
      MockWebSocket.instances[0].simulateClose()
    })

    await waitFor(() => {
      expect(result.current.status).toBe('disconnected')
    })
  })

  it('updates status to error on error', async () => {
    const { result } = renderHook(() => useWebSocket('signals'))

    act(() => {
      MockWebSocket.instances[0].simulateError()
    })

    await waitFor(() => {
      expect(result.current.status).toBe('error')
    })
  })

  it('parses and stores last message', async () => {
    const { result } = renderHook(() => useWebSocket('signals'))

    act(() => {
      MockWebSocket.instances[0].simulateOpen()
    })

    const testMessage = { type: 'signal', data: { id: 'SIG-001' } }
    act(() => {
      MockWebSocket.instances[0].simulateMessage(testMessage)
    })

    await waitFor(() => {
      expect(result.current.lastMessage).toEqual(testMessage)
    })
  })

  it('calls onMessage callback when message received', async () => {
    const onMessage = vi.fn()
    renderHook(() => useWebSocket('signals', { onMessage }))

    act(() => {
      MockWebSocket.instances[0].simulateOpen()
    })

    const testMessage = { type: 'signal', data: { id: 'SIG-001' } }
    act(() => {
      MockWebSocket.instances[0].simulateMessage(testMessage)
    })

    await waitFor(() => {
      expect(onMessage).toHaveBeenCalledWith(testMessage)
    })
  })

  it('calls onClose callback when connection closes', async () => {
    const onClose = vi.fn()
    renderHook(() => useWebSocket('signals', { onClose }))

    act(() => {
      MockWebSocket.instances[0].simulateOpen()
      MockWebSocket.instances[0].simulateClose()
    })

    await waitFor(() => {
      expect(onClose).toHaveBeenCalled()
    })
  })

  it('sendMessage sends data when connected', async () => {
    const { result } = renderHook(() => useWebSocket('signals'))

    act(() => {
      MockWebSocket.instances[0].simulateOpen()
    })

    await waitFor(() => {
      expect(result.current.status).toBe('connected')
    })

    act(() => {
      result.current.sendMessage('test message')
    })

    expect(MockWebSocket.instances[0].send).toHaveBeenCalledWith('test message')
  })

  it('sendMessage stringifies objects', async () => {
    const { result } = renderHook(() => useWebSocket('signals'))

    act(() => {
      MockWebSocket.instances[0].simulateOpen()
    })

    await waitFor(() => {
      expect(result.current.status).toBe('connected')
    })

    act(() => {
      result.current.sendMessage({ type: 'subscribe' })
    })

    expect(MockWebSocket.instances[0].send).toHaveBeenCalledWith('{"type":"subscribe"}')
  })

  it('ping sends ping message', async () => {
    const { result } = renderHook(() => useWebSocket('signals'))

    act(() => {
      MockWebSocket.instances[0].simulateOpen()
    })

    await waitFor(() => {
      expect(result.current.status).toBe('connected')
    })

    act(() => {
      result.current.ping()
    })

    expect(MockWebSocket.instances[0].send).toHaveBeenCalledWith('ping')
  })

  it('disconnect closes connection', async () => {
    const { result } = renderHook(() => useWebSocket('signals'))

    act(() => {
      MockWebSocket.instances[0].simulateOpen()
    })

    act(() => {
      result.current.disconnect()
    })

    expect(MockWebSocket.instances[0].close).toHaveBeenCalled()
    expect(result.current.status).toBe('disconnected')
  })

  it('cleans up WebSocket on unmount', () => {
    const { unmount } = renderHook(() => useWebSocket('signals'))

    act(() => {
      MockWebSocket.instances[0].simulateOpen()
    })

    unmount()

    expect(MockWebSocket.instances[0].close).toHaveBeenCalled()
  })

  it('uses different endpoints', () => {
    renderHook(() => useWebSocket('prices'))

    expect(MockWebSocket.instances[0].url).toContain('/ws/prices')
  })
})
