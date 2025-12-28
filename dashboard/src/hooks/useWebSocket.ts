import { useEffect, useRef, useState, useCallback } from 'react'
import type { WSMessage } from '@/types'

type WebSocketStatus = 'connecting' | 'connected' | 'disconnected' | 'error'

interface UseWebSocketOptions {
  onMessage?: (message: WSMessage) => void
  onError?: (error: Event) => void
  onClose?: () => void
  reconnectAttempts?: number
  reconnectInterval?: number
}

export function useWebSocket(
  endpoint: 'signals' | 'prices' | 'portfolio',
  options: UseWebSocketOptions = {}
) {
  const {
    onMessage,
    onError,
    onClose,
    reconnectAttempts = 5,
    reconnectInterval = 3000,
  } = options

  const [status, setStatus] = useState<WebSocketStatus>('disconnected')
  const [lastMessage, setLastMessage] = useState<WSMessage | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectCountRef = useRef(0)
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const connect = useCallback(() => {
    const token = localStorage.getItem('api_key') || ''
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const host = window.location.host
    const url = `${protocol}//${host}/ws/${endpoint}?token=${token}`

    setStatus('connecting')

    const ws = new WebSocket(url)
    wsRef.current = ws

    ws.onopen = () => {
      setStatus('connected')
      reconnectCountRef.current = 0
    }

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data) as WSMessage
        setLastMessage(message)
        onMessage?.(message)
      } catch {
        console.error('Failed to parse WebSocket message:', event.data)
      }
    }

    ws.onerror = (event) => {
      setStatus('error')
      onError?.(event)
    }

    ws.onclose = () => {
      setStatus('disconnected')
      onClose?.()

      // Attempt to reconnect
      if (reconnectCountRef.current < reconnectAttempts) {
        reconnectCountRef.current += 1
        reconnectTimeoutRef.current = setTimeout(() => {
          connect()
        }, reconnectInterval)
      }
    }
  }, [endpoint, onMessage, onError, onClose, reconnectAttempts, reconnectInterval])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    if (wsRef.current) {
      wsRef.current.close()
      wsRef.current = null
    }
    setStatus('disconnected')
  }, [])

  const sendMessage = useCallback((message: string | object) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      const data = typeof message === 'string' ? message : JSON.stringify(message)
      wsRef.current.send(data)
    }
  }, [])

  const ping = useCallback(() => {
    sendMessage('ping')
  }, [sendMessage])

  useEffect(() => {
    connect()
    return () => disconnect()
  }, [connect, disconnect])

  return {
    status,
    lastMessage,
    sendMessage,
    ping,
    connect,
    disconnect,
  }
}
