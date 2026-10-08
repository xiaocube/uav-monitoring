import { useState, useRef, useEffect, useCallback } from 'react'
import { io } from 'socket.io-client'
import './App.css'

const SERVER_URL = 'http://localhost:5001'

function App() {
  const [isDetecting, setIsDetecting] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [backendConnected, setBackendConnected] = useState(false)
  const [params, setParams] = useState({
    conf: 0.20,
    iou: 0.50,
    imgsz: 800,
    device: 'mps',
    tile_mode: false,
    filter_enabled: true,
    wbf_iou: 0.55,
  })
  const [detections, setDetections] = useState([])
  const [fps, setFps] = useState(0)
  const [modelInfo, setModelInfo] = useState(null)
  const [availableModels, setAvailableModels] = useState({ models: [], ensembles: [] })
  const [showSettings, setShowSettings] = useState(false)
  const [showModelSelector, setShowModelSelector] = useState(false)
  const [logMessages, setLogMessages] = useState([])
  const [useSimulated, setUseSimulated] = useState(true)
  const [testMode, setTestMode] = useState('webcam')
  const [isProcessingImage, setIsProcessingImage] = useState(false)

  const canvasRef = useRef(null)
  const wsRef = useRef(null)
  const imgRef = useRef(null)
  const fileInputRef = useRef(null)
  const animationRef = useRef(null)
  const lastFrameTimeRef = useRef(Date.now())
  const frameCountRef = useRef(0)
  const startDetectionSimulatedRef = useRef(() => {})

  const addLog = useCallback((message, type = 'info') => {
    const timestamp = new Date().toLocaleTimeString()
    setLogMessages(prev => [
      ...prev.slice(-50),
      { timestamp, message, type }
    ])
  }, [])

  const handleParamChange = useCallback((key, value) => {
    setParams(prev => ({ ...prev, [key]: value }))
  }, [])

  // 加载可用模型列表
  const fetchModels = useCallback(async () => {
    try {
      const response = await fetch(`${SERVER_URL}/api/models`)
      if (response.ok) {
        const data = await response.json()
        setAvailableModels(data)
        if (data.current) {
          setModelInfo(data.current)
        }
      }
    } catch (e) {
      addLog('获取模型列表失败', 'warning')
    }
  }, [addLog])

  const connectBackend = useCallback(() => {
    try {
      const socket = io(SERVER_URL, {
        transports: ['polling', 'websocket'],
        timeout: 5000,
        reconnection: true,
        reconnectionAttempts: 3,
        reconnectionDelay: 3000,
        reconnectionDelayMax: 10000,
      })

      socket.on('connect', () => {
        setBackendConnected(true)
        setUseSimulated(false)
        addLog('后端服务器连接成功', 'success')
        fetchModels()
      })

      socket.on('status', (data) => {
        if (data.params) {
          setParams(prev => ({ ...prev, ...data.params }))
        }
        if (data.model_info) {
          setModelInfo(data.model_info)
        }
        addLog(`模型状态: ${data.is_connected ? '已加载' : '未加载'}`, 'info')
      })

      socket.on('model_loaded', (data) => {
        if (data.success) {
          addLog(data.message, 'success')
          if (data.model_info) {
            setModelInfo(data.model_info)
          }
        } else {
          addLog(`加载失败: ${data.message}`, 'error')
        }
      })

      socket.on('frame', (data) => {
        const img = imgRef.current
        if (img && data.image) {
          img.src = `data:image/jpeg;base64,${data.image}`
          img.style.display = 'block'
          const canvas = canvasRef.current
          if (canvas) canvas.style.display = 'none'
        }
        if (data.detections) {
          setDetections(data.detections)
        }
        if (data.fps !== undefined) {
          setFps(data.fps)
        }
      })

      socket.on('error', (data) => {
        addLog(`错误: ${data.message}`, 'error')
      })

      socket.on('disconnect', () => {
        setBackendConnected(false)
        setUseSimulated(true)
        addLog('后端连接断开，切换到演示模式', 'warning')
      })

      let errorLogged = false
      socket.on('connect_error', () => {
        setBackendConnected(false)
        setUseSimulated(true)
        if (!errorLogged) {
          errorLogged = true
          addLog('后端未启动（端口5001），使用演示模式', 'warning')
        }
      })

      socket.io.on('reconnect_failed', () => {
        addLog('后端连接重试失败，请启动后端服务后刷新页面', 'error')
      })

      wsRef.current = socket
    } catch (e) {
      setUseSimulated(true)
      addLog('使用模拟演示模式', 'info')
    }
  }, [addLog, fetchModels])

  const loadModel = useCallback((modelId, type = 'single') => {
    const socket = wsRef.current
    if (!socket || !socket.connected) {
      addLog('请先连接后端', 'warning')
      return
    }

    if (type === 'single') {
      socket.emit('load_model', { model_id: modelId })
      addLog(`正在加载模型: ${modelId}...`, 'info')
    } else {
      socket.emit('load_ensemble', { ensemble_id: modelId })
      addLog(`正在加载集成: ${modelId}...`, 'info')
    }
  }, [addLog])

  // 图片检测
  const detectImage = useCallback(async (file) => {
    if (!modelInfo) {
      addLog('请先选择模型', 'warning')
      return
    }
    if (!backendConnected) {
      addLog('后端未连接，无法进行图片检测', 'error')
      return
    }

    setIsProcessingImage(true)
    addLog(`正在分析图片: ${file.name}`, 'info')

    try {
      const reader = new FileReader()
      reader.onload = async (e) => {
        const base64 = e.target.result
        try {
          const response = await fetch(`${SERVER_URL}/api/detect-image`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64 }),
          })
          if (response.ok) {
            const data = await response.json()
            if (data.success) {
              setDetections(data.detections)
              setFps(0)
              if (imgRef.current) {
                imgRef.current.src = `data:image/jpeg;base64,${data.image}`
                imgRef.current.style.display = 'block'
                const canvas = canvasRef.current
                if (canvas) canvas.style.display = 'none'
              }
              addLog(`检测完成，发现 ${data.count} 个目标`, 'success')
            } else {
              addLog(`检测失败: ${data.message}`, 'error')
            }
          } else {
            addLog('服务器响应错误', 'error')
          }
        } catch (err) {
          addLog(`请求错误: ${err.message}`, 'error')
        } finally {
          setIsProcessingImage(false)
        }
      }
      reader.onerror = () => {
        addLog('文件读取失败', 'error')
        setIsProcessingImage(false)
      }
      reader.readAsDataURL(file)
    } catch (err) {
      addLog(`错误: ${err.message}`, 'error')
      setIsProcessingImage(false)
    }
  }, [modelInfo, backendConnected, addLog])

  const handleFileSelect = useCallback((e) => {
    const file = e.target.files?.[0]
    if (file) {
      detectImage(file)
    }
    e.target.value = ''
  }, [detectImage])

  const loadSampleImage = useCallback(async (sampleName) => {
    addLog(`加载示例图片: ${sampleName}`, 'info')
    try {
      const response = await fetch(`/samples/${sampleName}`)
      if (!response.ok) throw new Error('加载失败')
      const blob = await response.blob()
      const file = new File([blob], sampleName, { type: 'image/jpeg' })
      detectImage(file)
    } catch (err) {
      addLog(`加载失败: ${err.message}`, 'error')
    }
  }, [addLog, detectImage])

  const startDetectionReal = useCallback(() => {
    if (!modelInfo) {
      addLog('请先选择模型', 'warning')
      return
    }

    addLog('正在启动真实检测...', 'info')
    const socket = wsRef.current
    if (socket && socket.connected) {
      socket.emit('start_detection', { params }, (response) => {
        if (response && response.success) {
          setIsDetecting(true)
          addLog('真实检测已启动（摄像头）', 'success')
        } else {
          addLog(`启动失败: ${response?.message || '未知错误'}`, 'error')
          setUseSimulated(true)
        }
      })
    } else {
      addLog('后端未连接，使用演示模式', 'warning')
      setUseSimulated(true)
      startDetectionSimulatedRef.current()
    }
  }, [addLog, params, modelInfo])

  const startDetectionSimulated = useCallback(() => {
    if (isDetecting) return

    addLog('正在启动检测系统（演示模式）...', 'info')
    setIsDetecting(true)
    addLog('检测系统已启动', 'success')

    let x = 100
    let y = 100
    let dx = 2
    let dy = 1.5

    const animate = () => {
      const canvas = canvasRef.current
      if (!canvas) return

      const ctx = canvas.getContext('2d')
      const w = canvas.width
      const h = canvas.height

      ctx.fillStyle = '#0a0f1a'
      ctx.fillRect(0, 0, w, h)

      ctx.strokeStyle = 'rgba(34, 197, 94, 0.05)'
      ctx.lineWidth = 1
      for (let i = 0; i < w; i += 40) {
        ctx.beginPath()
        ctx.moveTo(i, 0)
        ctx.lineTo(i, h)
        ctx.stroke()
      }
      for (let i = 0; i < h; i += 40) {
        ctx.beginPath()
        ctx.moveTo(0, i)
        ctx.lineTo(w, i)
        ctx.stroke()
      }

      ctx.fillStyle = 'rgba(255, 255, 255, 0.03)'
      ctx.beginPath()
      ctx.moveTo(0, h * 0.7)
      ctx.lineTo(w * 0.3, h * 0.5)
      ctx.lineTo(w * 0.5, h * 0.6)
      ctx.lineTo(w * 0.7, h * 0.45)
      ctx.lineTo(w, h * 0.55)
      ctx.lineTo(w, h)
      ctx.lineTo(0, h)
      ctx.closePath()
      ctx.fill()

      x += dx
      y += dy
      if (x > w - 80 || x < 20) dx = -dx
      if (y > h - 150 || y < 20) dy = -dy

      const newDetections = [
        {
          id: 1,
          class: 'uav',
          conf: (0.85 + Math.random() * 0.12),
          x: x,
          y: y,
          w: 80,
          h: 50,
        },
        {
          id: 2,
          class: 'uav',
          conf: (0.72 + Math.random() * 0.15),
          x: x + 150,
          y: y + 80,
          w: 60,
          h: 40,
        },
      ]

      setDetections(newDetections)

      newDetections.forEach(det => {
        ctx.strokeStyle = '#22c55e'
        ctx.lineWidth = 2
        ctx.strokeRect(det.x, det.y, det.w, det.h)

        ctx.fillStyle = 'rgba(34, 197, 94, 0.9)'
        const label = `${det.class} ${(det.conf * 100).toFixed(0)}%`
        ctx.font = '12px monospace'
        const textW = ctx.measureText(label).width
        ctx.fillRect(det.x, det.y - 20, textW + 8, 20)
        ctx.fillStyle = '#fff'
        ctx.fillText(label, det.x + 4, det.y - 6)

        ctx.fillStyle = '#22c55e'
        ctx.beginPath()
        ctx.arc(det.x + det.w / 2, det.y + det.h / 2, 4, 0, Math.PI * 2)
        ctx.fill()
      })

      frameCountRef.current++
      const now = Date.now()
      if (now - lastFrameTimeRef.current >= 1000) {
        setFps(frameCountRef.current)
        frameCountRef.current = 0
        lastFrameTimeRef.current = now
      }

      animationRef.current = requestAnimationFrame(animate)
    }

    animationRef.current = requestAnimationFrame(animate)
  }, [isDetecting, addLog])

  startDetectionSimulatedRef.current = startDetectionSimulated

  const startDetection = useCallback(() => {
    if (!useSimulated && backendConnected) {
      startDetectionReal()
    } else {
      startDetectionSimulated()
    }
  }, [useSimulated, backendConnected, startDetectionReal, startDetectionSimulated])

  const stopDetection = useCallback(() => {
    if (animationRef.current) {
      cancelAnimationFrame(animationRef.current)
      animationRef.current = null
    }
    const socket = wsRef.current
    if (socket && socket.connected) {
      socket.emit('stop_detection')
    }
    const img = imgRef.current
    if (img) img.style.display = 'none'
    const canvas = canvasRef.current
    if (canvas) canvas.style.display = 'block'

    setIsDetecting(false)
    setDetections([])
    setFps(0)
    addLog('检测系统已停止', 'warning')
  }, [addLog])

  useEffect(() => {
    const socket = wsRef.current
    if (socket && socket.connected) {
      socket.emit('update_params', { params })
    }
  }, [params])

  const resetParams = useCallback(() => {
    setParams({
      conf: 0.20,
      iou: 0.50,
      imgsz: 800,
      device: 'mps',
      tile_mode: false,
      filter_enabled: true,
      wbf_iou: 0.55,
    })
    addLog('参数已重置为默认值', 'info')
  }, [addLog])

  useEffect(() => {
    addLog('系统初始化完成', 'success')
    setIsConnected(true)
    connectBackend()

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      if (wsRef.current) {
        wsRef.current.disconnect()
      }
    }
  }, [addLog, connectBackend])

  return (
    <div className="flex h-screen bg-[#0f1520] text-white overflow-hidden">
      <div className="flex-1 flex flex-col">
        {/* 顶部栏 */}
        <header className="h-14 border-b border-white/10 bg-[#1a2332] flex items-center justify-between px-6 shrink-0">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-400 to-accent-600 flex items-center justify-center">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
              </svg>
            </div>
            <div>
              <h1 className="text-lg font-semibold">SkyGuard 无人机监测系统</h1>
              <p className="text-xs text-gray-400">
                {modelInfo ? `${modelInfo.modelName} | ${modelInfo.architecture}` : 'v2.0.0'}
              </p>
            </div>
          </div>

          <div className="flex items-center gap-4">
            <button
              onClick={() => setShowModelSelector(!showModelSelector)}
              className={`px-3 py-1.5 rounded-lg text-sm transition-all flex items-center gap-2 ${
                showModelSelector
                  ? 'bg-accent-500 text-white'
                  : 'bg-white/10 text-gray-300 hover:bg-white/20'
              }`}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
              </svg>
              选择模型
            </button>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500 animate-pulse' : 'bg-red-500'}`}></div>
              <span className="text-sm text-gray-300">{isConnected ? '已连接' : '未连接'}</span>
            </div>
            <div className="h-6 w-px bg-white/10"></div>
            <div className="text-right">
              <p className="text-xs text-gray-400">FPS</p>
              <p className="text-sm font-mono font-bold text-accent-400">{fps}</p>
            </div>
          </div>
        </header>

        {/* 主内容区 */}
        <div className="flex-1 flex min-h-0">
          {/* 左侧视频区 */}
          <div className="flex-1 p-4">
            <div className="relative w-full h-full rounded-xl overflow-hidden bg-[#0a0f1a] border border-white/10">
              <canvas
                ref={canvasRef}
                width={1280}
                height={720}
                className="absolute inset-0 w-full h-full object-contain"
              />
              <img
                ref={imgRef}
                className="hidden absolute inset-0 w-full h-full object-contain"
                alt="detection"
              />
              
              {/* 状态标签 */}
              <div className="absolute top-4 left-4 flex gap-2">
                {isDetecting && (
                  <div className="flex items-center gap-2 bg-red-500/90 px-3 py-1.5 rounded-lg text-sm font-medium">
                    <span className="w-2 h-2 bg-white rounded-full animate-pulse"></span>
                    REC
                  </div>
                )}
                <div className="bg-black/60 backdrop-blur px-3 py-1.5 rounded-lg text-sm">
                  检测目标: {detections.length}
                </div>
                {modelInfo?.mode === 'ensemble' && (
                  <div className="bg-yellow-500/20 backdrop-blur border border-yellow-500/50 px-3 py-1.5 rounded-lg text-xs text-yellow-400">
                    WBF 集成模式
                  </div>
                )}
                {useSimulated && (
                  <div className="bg-yellow-500/20 backdrop-blur border border-yellow-500/50 px-3 py-1.5 rounded-lg text-xs text-yellow-400">
                    演示模式
                  </div>
                )}
                {params.tile_mode && (
                  <div className="bg-blue-500/20 backdrop-blur border border-blue-500/50 px-3 py-1.5 rounded-lg text-xs text-blue-400">
                    切片推理 2x2
                  </div>
                )}
                {params.filter_enabled && (
                  <div className="bg-purple-500/20 backdrop-blur border border-purple-500/50 px-3 py-1.5 rounded-lg text-xs text-purple-400">
                    过滤 ON
                  </div>
                )}
              </div>

              {/* 参数面板 */}
              <div className="absolute top-4 right-4 bg-black/60 backdrop-blur px-3 py-2 rounded-lg text-xs font-mono">
                <div>分辨率: {params.imgsz}px</div>
                <div>设备: {params.device.toUpperCase()}</div>
                <div>置信度: {params.conf.toFixed(2)}</div>
                {modelInfo?.mode === 'ensemble' && (
                  <div>WBF IoU: {params.wbf_iou.toFixed(2)}</div>
                )}
                <div>过滤: {params.filter_enabled ? 'ON' : 'OFF'}</div>
              </div>

              {/* 未启动时的提示 */}
              {!isDetecting && testMode === 'webcam' && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/40">
                  <div className="w-20 h-20 rounded-full bg-white/5 flex items-center justify-center mb-4">
                    <svg className="w-10 h-10 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <p className="text-gray-400 mb-1">点击开始检测按钮启动视频检测</p>
                  <p className="text-sm text-gray-500">支持摄像头输入 / 视频文件</p>
                  {!modelInfo && (
                    <p className="text-sm text-yellow-400 mt-2">请先在右侧选择一个模型</p>
                  )}
                </div>
              )}

              {testMode === 'image' && !imgRef.current?.src && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-black/40">
                  <div className="w-20 h-20 rounded-full bg-white/5 flex items-center justify-center mb-4">
                    <svg className="w-10 h-10 text-gray-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                    </svg>
                  </div>
                  <p className="text-gray-400 mb-1">上传图片或选择示例图片进行测试</p>
                  <p className="text-sm text-gray-500">支持 JPG / PNG 格式</p>
                  {!modelInfo && (
                    <p className="text-sm text-yellow-400 mt-2">请先在右侧选择一个模型</p>
                  )}
                </div>
              )}

              {/* 测试模式切换 */}
              <div className="absolute bottom-4 left-4 bg-black/60 backdrop-blur rounded-lg p-1 flex gap-1">
                <button
                  onClick={() => { stopDetection(); setTestMode('webcam') }}
                  className={`px-3 py-1.5 text-xs rounded transition-all ${
                    testMode === 'webcam' ? 'bg-accent-500 text-white' : 'text-gray-400 hover:text-white'
                  }`}
                >
                  📹 摄像头
                </button>
                <button
                  onClick={() => { stopDetection(); setTestMode('image') }}
                  className={`px-3 py-1.5 text-xs rounded transition-all ${
                    testMode === 'image' ? 'bg-accent-500 text-white' : 'text-gray-400 hover:text-white'
                  }`}
                >
                  🖼️ 图片测试
                </button>
              </div>
            </div>
          </div>

          {/* 右侧控制面板 */}
          <div className="w-80 border-l border-white/10 bg-[#1a2332] flex flex-col shrink-0">
            {/* 模型选择面板 */}
            {showModelSelector && (
              <div className="p-4 border-b border-white/10 max-h-96 overflow-y-auto">
                <div className="flex justify-between items-center mb-3">
                  <h3 className="text-sm font-semibold text-gray-300">模型选择</h3>
                  <button
                    onClick={() => setShowModelSelector(false)}
                    className="text-gray-400 hover:text-white text-xs"
                  >
                    收起
                  </button>
                </div>

                {/* 集成方案 */}
                {availableModels.ensembles?.length > 0 && (
                  <div className="mb-4">
                    <p className="text-xs text-gray-500 mb-2 font-medium">集成方案 (WBF)</p>
                    <div className="space-y-2">
                      {availableModels.ensembles.map(ens => (
                        <div
                          key={ens.id}
                          onClick={() => loadModel(ens.id, 'ensemble')}
                          className={`p-2.5 rounded-lg cursor-pointer transition-all border ${
                            modelInfo?.modelId === ens.id
                              ? 'bg-accent-500/20 border-accent-500'
                              : 'bg-white/5 border-white/10 hover:bg-white/10'
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-white">{ens.name}</span>
                            <span className="text-xs bg-yellow-500/20 text-yellow-400 px-1.5 py-0.5 rounded">
                              WBF
                            </span>
                          </div>
                          <p className="text-xs text-gray-400 mt-1">{ens.description}</p>
                          <div className="flex gap-1 mt-1.5">
                            {ens.models.map(m => (
                              <span key={m} className="text-xs bg-white/10 px-1.5 py-0.5 rounded text-gray-400">
                                {m}
                              </span>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* 单模型 */}
                {availableModels.models?.length > 0 && (
                  <div>
                    <p className="text-xs text-gray-500 mb-2 font-medium">单模型</p>
                    <div className="space-y-2">
                      {availableModels.models.map(model => (
                        <div
                          key={model.id}
                          onClick={() => model.available && loadModel(model.id, 'single')}
                          className={`p-2.5 rounded-lg cursor-pointer transition-all border ${
                            !model.available
                              ? 'bg-red-500/10 border-red-500/30 opacity-50 cursor-not-allowed'
                              : modelInfo?.modelId === model.id
                                ? 'bg-accent-500/20 border-accent-500'
                                : 'bg-white/5 border-white/10 hover:bg-white/10'
                          }`}
                        >
                          <div className="flex items-center justify-between">
                            <span className="text-sm font-medium text-white">{model.name}</span>
                            {!model.available && (
                              <span className="text-xs text-red-400">文件缺失</span>
                            )}
                            {model.tags?.includes('recommended') && (
                              <span className="text-xs bg-green-500/20 text-green-400 px-1.5 py-0.5 rounded">
                                推荐
                              </span>
                            )}
                          </div>
                          <div className="flex gap-3 mt-1 text-xs text-gray-400">
                            <span>{model.architecture}</span>
                            <span>mAP50: {(model.mAP50 * 100).toFixed(1)}%</span>
                            {model.fps > 0 && <span>FPS: {model.fps}</span>}
                          </div>
                          <p className="text-xs text-gray-500 mt-1">{model.description}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* 控制面板 */}
            <div className="p-4 border-b border-white/10">
              <h3 className="text-sm font-semibold text-gray-300 mb-3">控制面板</h3>

              {/* 图片测试模式 */}
              {testMode === 'image' && (
                <div className="mb-3 space-y-2">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    disabled={!modelInfo || isProcessingImage}
                    className="w-full py-2.5 px-4 rounded-lg bg-gradient-to-r from-accent-500 to-accent-600 hover:from-accent-600 hover:to-accent-700 text-white font-medium text-sm transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                  >
                    {isProcessingImage ? (
                      <>
                        <span className="w-2 h-2 bg-white rounded-full animate-pulse"></span>
                        分析中...
                      </>
                    ) : (
                      <>
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                        </svg>
                        上传图片
                      </>
                    )}
                  </button>
                  <div className="pt-2 border-t border-white/10">
                    <p className="text-xs text-gray-500 mb-1.5">示例图片</p>
                    <div className="grid grid-cols-3 gap-1.5">
                      {['demo_01_1uav.jpg', 'demo_02_2uav.jpg', 'demo_03_1uav.jpg', 'demo_04_1uav.jpg', 'demo_05_1uav.jpg', 'demo_06_1uav.jpg'].map(name => (
                        <button
                          key={name}
                          onClick={() => loadSampleImage(name)}
                          disabled={!modelInfo || isProcessingImage}
                          className="text-xs py-1.5 px-1 rounded bg-white/5 hover:bg-white/10 border border-white/10 text-gray-400 hover:text-white transition-all truncate disabled:opacity-50 disabled:cursor-not-allowed"
                          title={name}
                        >
                          {name.replace('demo_', '').replace('.jpg', '')}
                        </button>
                      ))}
                    </div>
                  </div>
                </div>
              )}
              
              <div className="flex gap-2">
                {testMode === 'webcam' ? (
                  <button
                    onClick={isDetecting ? stopDetection : startDetection}
                    disabled={!modelInfo && !useSimulated}
                    className={`flex-1 py-2.5 px-4 rounded-lg font-medium text-sm transition-all ${
                      isDetecting
                        ? 'bg-red-500 hover:bg-red-600 text-white'
                        : (!modelInfo && !useSimulated)
                          ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                          : 'bg-accent-500 hover:bg-accent-600 text-white'
                    }`}
                  >
                    {isDetecting ? '停止检测' : '开始检测'}
                  </button>
                ) : (
                  <button
                    onClick={() => {
                      if (imgRef.current) {
                        imgRef.current.src = ''
                        imgRef.current.style.display = 'none'
                      }
                      setDetections([])
                      addLog('已清空检测结果', 'info')
                    }}
                    className="flex-1 py-2.5 px-4 rounded-lg font-medium text-sm bg-white/5 hover:bg-white/10 text-gray-300 transition-all"
                  >
                    清空结果
                  </button>
                )}
                <button
                  onClick={() => setShowSettings(!showSettings)}
                  className={`p-2.5 rounded-lg border transition-all ${
                    showSettings
                      ? 'bg-accent-500/20 border-accent-500 text-accent-400'
                      : 'border-white/10 text-gray-400 hover:border-white/20 hover:text-gray-300'
                  }`}
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </button>
              </div>
            </div>

            {/* 参数设置 */}
            {showSettings && (
              <div className="p-4 border-b border-white/10 space-y-4">
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <label className="text-sm text-gray-300">置信度阈值</label>
                    <span className="text-sm font-mono text-accent-400">{params.conf.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={params.conf}
                    onChange={(e) => handleParamChange('conf', parseFloat(e.target.value))}
                    className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                <div>
                  <div className="flex justify-between items-center mb-2">
                    <label className="text-sm text-gray-300">IOU 阈值</label>
                    <span className="text-sm font-mono text-accent-400">{params.iou.toFixed(2)}</span>
                  </div>
                  <input
                    type="range"
                    min="0.1"
                    max="0.9"
                    step="0.05"
                    value={params.iou}
                    onChange={(e) => handleParamChange('iou', parseFloat(e.target.value))}
                    className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                {/* WBF IOU - 仅集成模式显示 */}
                {modelInfo?.mode === 'ensemble' && (
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm text-gray-300">WBF 融合 IOU</label>
                      <span className="text-sm font-mono text-accent-400">{params.wbf_iou.toFixed(2)}</span>
                    </div>
                    <input
                      type="range"
                      min="0.3"
                      max="0.8"
                      step="0.05"
                      value={params.wbf_iou}
                      onChange={(e) => handleParamChange('wbf_iou', parseFloat(e.target.value))}
                      className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>
                )}

                <div>
                  <div className="flex justify-between items-center mb-2">
                    <label className="text-sm text-gray-300">输入分辨率</label>
                    <span className="text-sm font-mono text-accent-400">{params.imgsz}px</span>
                  </div>
                  <input
                    type="range"
                    min="320"
                    max="1280"
                    step="32"
                    value={params.imgsz}
                    onChange={(e) => handleParamChange('imgsz', parseInt(e.target.value))}
                    className="w-full h-2 bg-white/10 rounded-lg appearance-none cursor-pointer"
                  />
                </div>

                <div>
                  <label className="text-sm text-gray-300 block mb-2">推理设备</label>
                  <select
                    value={params.device}
                    onChange={(e) => handleParamChange('device', e.target.value)}
                    className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-2 text-sm text-white focus:outline-none focus:border-accent-500"
                  >
                    <option value="cpu">CPU</option>
                    <option value="mps">MPS (Apple Silicon)</option>
                    <option value="cuda">CUDA (NVIDIA GPU)</option>
                  </select>
                </div>

                {/* 后处理过滤开关 */}
                <div className="flex items-center justify-between bg-white/5 rounded-lg px-3 py-2.5">
                  <div>
                    <span className="text-sm text-gray-300">后处理过滤</span>
                    <p className="text-xs text-gray-500 mt-0.5">尺寸/宽高比/置信度过滤误报</p>
                  </div>
                  <button
                    onClick={() => {
                      const newVal = !params.filter_enabled
                      handleParamChange('filter_enabled', newVal)
                      addLog(`后处理过滤已${newVal ? '开启' : '关闭'}`, newVal ? 'success' : 'warning')
                    }}
                    className={`relative w-11 h-6 rounded-full transition-colors shrink-0 ${
                      params.filter_enabled ? 'bg-accent-500' : 'bg-white/10'
                    }`}
                  >
                    <span className={`absolute top-0.5 left-0.5 w-5 h-5 rounded-full bg-white transition-transform ${
                      params.filter_enabled ? 'translate-x-5' : ''
                    }`} />
                  </button>
                </div>

                <button
                  onClick={resetParams}
                  className="w-full py-2 px-4 rounded-lg border border-white/10 text-sm text-gray-300 hover:bg-white/5 transition-colors"
                >
                  重置参数
                </button>
              </div>
            )}

            {/* 当前模型信息 */}
            {modelInfo && (
              <div className="p-4 border-b border-white/10">
                <h3 className="text-sm font-semibold text-gray-300 mb-3">当前模型</h3>
                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500">模式</span>
                    <span className={`font-medium ${modelInfo.mode === 'ensemble' ? 'text-yellow-400' : 'text-accent-400'}`}>
                      {modelInfo.mode === 'ensemble' ? 'WBF 集成' : '单模型'}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">模型名称</span>
                    <span className="text-gray-300 font-mono text-xs">{modelInfo.modelName}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">架构</span>
                    <span className="text-gray-300">{modelInfo.architecture}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">目标类别</span>
                    <span className="text-gray-300">{modelInfo.classes?.length || 1} 类</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">mAP50</span>
                    <span className="text-accent-400 font-medium">{modelInfo.mAP50}</span>
                  </div>
                  {modelInfo.subModels && (
                    <div>
                      <span className="text-gray-500 text-xs">子模型:</span>
                      <div className="flex flex-wrap gap-1 mt-1">
                        {modelInfo.subModels.map((m, i) => (
                          <span key={i} className="text-xs bg-white/10 px-1.5 py-0.5 rounded text-gray-400">
                            {m}
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  {modelInfo.description && (
                    <p className="text-xs text-gray-500 mt-1">{modelInfo.description}</p>
                  )}
                </div>
              </div>
            )}

            {/* 检测结果 */}
            {detections.length > 0 && (
              <div className="p-4 border-b border-white/10 max-h-40 overflow-y-auto">
                <h3 className="text-sm font-semibold text-gray-300 mb-3">检测结果 ({detections.length})</h3>
                <div className="space-y-2">
                  {detections.slice(0, 5).map((det, idx) => (
                    <div key={idx} className="bg-white/5 rounded-lg p-2.5">
                      <div className="flex justify-between items-center mb-1">
                        <span className="text-sm font-medium text-accent-400">
                          {det.class.toUpperCase()} #{det.id}
                          {det.modelCount > 1 && (
                            <span className="text-yellow-400 ml-1">[{det.modelCount}M]</span>
                          )}
                        </span>
                        <span className="text-xs font-mono bg-accent-500/20 text-accent-400 px-2 py-0.5 rounded">
                          {(det.conf * 100).toFixed(1)}%
                        </span>
                      </div>
                      <div className="text-xs text-gray-500 font-mono">
                        X:{Math.round(det.x)} Y:{Math.round(det.y)} W:{Math.round(det.w)} H:{Math.round(det.h)}
                      </div>
                    </div>
                  ))}
                  {detections.length > 5 && (
                    <p className="text-xs text-gray-500 text-center">...还有 {detections.length - 5} 个目标</p>
                  )}
                </div>
              </div>
            )}

            {/* 系统日志 */}
            <div className="flex-1 flex flex-col min-h-0 p-4">
              <h3 className="text-sm font-semibold text-gray-300 mb-3">系统日志</h3>
              <div className="flex-1 overflow-y-auto scrollbar-thin bg-black/30 rounded-lg p-3 font-mono text-xs space-y-1">
                {logMessages.map((log, idx) => (
                  <div key={idx} className="flex gap-2">
                    <span className="text-gray-600 shrink-0">{log.timestamp}</span>
                    <span className={`${
                      log.type === 'success' ? 'text-green-400' :
                      log.type === 'warning' ? 'text-yellow-400' :
                      log.type === 'error' ? 'text-red-400' :
                      'text-gray-400'
                    }`}>
                      {log.message}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
