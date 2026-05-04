import { useState, useEffect, useRef } from 'react'
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, LineChart, Line,
} from 'recharts'
import {
  AlertTriangle, Shield, Activity,
  Eye, ChevronDown, ChevronUp, Play, Square
} from 'lucide-react'

const API = 'http://localhost:8000'
const WS  = 'ws://localhost:8000/ws'

const S = {
  app: {
    minHeight: '100vh',
    background: '#0f1117',
    color: '#e2e8f0',
    fontFamily: '-apple-system, BlinkMacSystemFont, Segoe UI, sans-serif',
  },
  header: {
    background: '#1a1d27',
    borderBottom: '1px solid #2d3748',
    padding: '16px 32px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  headerLeft: { display: 'flex', alignItems: 'center', gap: 12 },
  headerTitle: { fontSize: 20, fontWeight: 700, color: '#fff' },
  headerSub: { fontSize: 13, color: '#718096', marginTop: 2 },
  main: { padding: '24px 32px', maxWidth: 1400, margin: '0 auto' },
  grid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(4, 1fr)',
    gap: 16,
    marginBottom: 24,
  },
  card: {
    background: '#1a1d27',
    border: '1px solid #2d3748',
    borderRadius: 12,
    padding: 20,
  },
  cardTitle: {
    fontSize: 12, color: '#718096',
    textTransform: 'uppercase',
    letterSpacing: '0.05em', marginBottom: 8,
  },
  cardValue: { fontSize: 28, fontWeight: 700, color: '#fff' },
  cardSub: { fontSize: 12, color: '#48bb78', marginTop: 4 },
  twoCol: {
    display: 'grid',
    gridTemplateColumns: '1fr 1fr',
    gap: 16, marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 15, fontWeight: 600, color: '#fff',
    marginBottom: 16, display: 'flex',
    alignItems: 'center', gap: 8,
  },
  alertItem: {
    background: '#1a1d27',
    border: '1px solid #2d3748',
    borderRadius: 10, padding: 16,
    marginBottom: 12, cursor: 'pointer',
    transition: 'border-color 0.2s',
  },
  narrative: {
    fontSize: 12, color: '#a0aec0',
    marginTop: 10, lineHeight: 1.6,
    borderLeft: '3px solid #3d4251',
    paddingLeft: 10,
  },
}

const severityColor = {
  CRITICAL: '#fc8181', HIGH: '#f6ad55',
  MEDIUM: '#fbd38d', LOW: '#63b3ed', INFO: '#68d391',
}

const badge = (severity) => ({
  padding: '2px 10px', borderRadius: 20,
  fontSize: 11, fontWeight: 700,
  background: severity === 'CRITICAL' ? '#742a2a' :
              severity === 'HIGH'     ? '#7b341e' :
              severity === 'MEDIUM'   ? '#744210' : '#1a365d',
  color: severityColor[severity] || '#63b3ed',
})

const ABLATION_DATA = [
  { name: 'Baseline',     f1: 0.7645 },
  { name: '+ Peer (NC1)', f1: 0.8055 },
  { name: '+ Auth',       f1: 0.8092 },
  { name: '+ Session',    f1: 0.7974 },
  { name: '+ Network',    f1: 0.7967 },
  { name: 'Full Model',   f1: 0.8215 },
]

const FEDERATED_ITEMS = [
  { label: 'Fed F1',     value: '0.8089', color: '#68d391' },
  { label: 'Central F1', value: '0.7893', color: '#63b3ed' },
  { label: 'Gap',        value: '1.96%',  color: '#f6ad55' },
  { label: 'ε spent',    value: '0.119',  color: '#fc8181' },
  { label: 'ε budget',   value: '0.5',    color: '#a0aec0' },
  { label: 'Rounds',     value: '2',      color: '#68d391' },
]

function SHAPBar({ features }) {
  if (!features || !features.length) return null
  const max = Math.max(...features.map(f => Math.abs(f.shap_value)), 1)
  return (
    <div style={{ marginTop: 12 }}>
      <div style={{ fontSize: 11, color: '#718096', marginBottom: 6 }}>
        SHAP Feature Contributions
      </div>
      {features.map((f, i) => (
        <div key={i} style={{
          display: 'flex', alignItems: 'center',
          gap: 10, marginBottom: 6,
        }}>
          <div style={{
            fontSize: 11, color: '#a0aec0',
            width: 200, textAlign: 'right', flexShrink: 0,
          }}>
            {f.feature.replace(/_/g, ' ')}
          </div>
          <div style={{
            height: 16,
            width: `${Math.abs(f.shap_value) / max * 160}px`,
            background: f.shap_value > 0 ? '#e53e3e' : '#3182ce',
            borderRadius: 3, minWidth: 2,
          }} />
          <div style={{ fontSize: 11, color: '#718096', width: 55 }}>
            {f.shap_value > 0 ? '+' : ''}{f.shap_value.toFixed(3)}
          </div>
        </div>
      ))}
    </div>
  )
}

function AlertCard({ alert }) {
  const [expanded, setExpanded] = useState(false)
  const [action, setAction] = useState(alert.analyst_action)

  const handleAction = async (decision) => {
    await fetch(`${API}/alerts/${alert.alert_id}/action`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ decision }),
    })
    setAction(decision)
  }

  return (
    <div
      style={{
        ...S.alertItem,
        borderColor: expanded
          ? severityColor[alert.severity] : '#2d3748',
      }}
      onClick={() => setExpanded(e => !e)}
    >
      <div style={{
        display: 'flex', alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <AlertTriangle
            size={16} color={severityColor[alert.severity]}
          />
          <span style={{
            fontSize: 13, fontWeight: 600, color: '#fff',
          }}>
            {alert.alert_id}
          </span>
          <span style={badge(alert.severity)}>{alert.severity}</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <span style={{ fontSize: 13, color: '#a0aec0' }}>
            Score:{' '}
            <strong style={{ color: '#fff' }}>
              {alert.ensemble_score?.toFixed(3)}
            </strong>
          </span>
          {expanded
            ? <ChevronUp size={14} color="#718096" />
            : <ChevronDown size={14} color="#718096" />
          }
        </div>
      </div>

      {expanded && (
        <div onClick={e => e.stopPropagation()}>
          {alert.shap_narrative && (
            <div style={S.narrative}>{alert.shap_narrative}</div>
          )}
          <SHAPBar features={alert.top_shap_features} />

          <div style={{
            display: 'flex', gap: 8, marginTop: 12,
          }}>
            {['confirm', 'dismiss', 'escalate'].map(dec => (
              <button
                key={dec}
                onClick={() => handleAction(dec)}
                style={{
                  padding: '4px 14px',
                  borderRadius: 6, border: 'none',
                  fontSize: 11, cursor: 'pointer',
                  fontWeight: action === dec ? 700 : 400,
                  background: action === dec
                    ? dec === 'confirm'  ? '#276749'
                    : dec === 'escalate' ? '#742a2a'
                    : '#2d3748'
                    : '#2d3748',
                  color: action === dec ? '#fff' : '#a0aec0',
                }}
              >
                {dec.toUpperCase()}
              </button>
            ))}
            <span style={{
              marginLeft: 'auto', fontSize: 11, color: '#4a5568',
              alignSelf: 'center',
            }}>
              {new Date(alert.timestamp).toLocaleTimeString()}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

export default function App() {
  const [alerts, setAlerts]       = useState([])
  const [stats, setStats]         = useState({})
  const [metrics, setMetrics]     = useState({ models: [] })
  const [activeTab, setActiveTab] = useState('alerts')
  const [simRunning, setSimRunning] = useState(false)
  const [wsStatus, setWsStatus]   = useState('disconnected')
  const wsRef = useRef(null)

  // WebSocket connection
  useEffect(() => {
    const connect = () => {
      const ws = new WebSocket(WS)
      wsRef.current = ws

      ws.onopen = () => {
        setWsStatus('connected')
      }

      ws.onmessage = (e) => {
        const msg = JSON.parse(e.data)
        if (msg.type === 'history') {
          setAlerts(msg.alerts.reverse())
          setStats(msg.stats)
        } else if (msg.type === 'new_alert') {
          setAlerts(prev => [msg.alert, ...prev].slice(0, 50))
          setStats(prev => ({
            ...prev,
            total: (prev.total || 0) + 1,
            [msg.alert.severity.toLowerCase()]:
              (prev[msg.alert.severity.toLowerCase()] || 0) + 1,
          }))
        }
      }

      ws.onclose = () => {
        setWsStatus('disconnected')
        setTimeout(connect, 3000)
      }

      ws.onerror = () => ws.close()
    }

    connect()

    // Load metrics
    fetch(`${API}/metrics`)
      .then(r => r.json())
      .then(setMetrics)
      .catch(() => {})

    // Ping keepalive
    const ping = setInterval(() => {
      if (wsRef.current?.readyState === 1) {
        wsRef.current.send('ping')
      }
    }, 20000)

    return () => {
      clearInterval(ping)
      wsRef.current?.close()
    }
  }, [])

  const toggleSimulator = async () => {
    if (simRunning) {
      await fetch(`${API}/simulator/stop`, { method: 'POST' })
      setSimRunning(false)
    } else {
      await fetch(
        `${API}/simulator/start?speed=1&max_events=500`,
        { method: 'POST' }
      )
      setSimRunning(true)
    }
  }

  const clearAlerts = async () => {
    await fetch(`${API}/alerts`, { method: 'DELETE' })
    setAlerts([])
    setStats({})
  }

  const tabs = ['alerts', 'models', 'federated', 'ablation']

  return (
    <div style={S.app}>
      {/* Header */}
      <div style={S.header}>
        <div style={S.headerLeft}>
          <Shield size={24} color="#63b3ed" />
          <div>
            <div style={S.headerTitle}>Behavioral Security Layer</div>
            <div style={S.headerSub}>
              Privacy-Preserving Explainable Anomaly Detection
            </div>
          </div>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <div style={{
            width: 8, height: 8, borderRadius: '50%',
            background: wsStatus === 'connected' ? '#68d391' : '#fc8181',
          }} />
          <span style={{ fontSize: 12, color: '#718096' }}>
            {wsStatus === 'connected' ? 'Live' : 'Reconnecting...'}
          </span>
          <button
            onClick={toggleSimulator}
            style={{
              display: 'flex', alignItems: 'center', gap: 6,
              padding: '6px 16px', borderRadius: 8,
              border: 'none', cursor: 'pointer', fontSize: 13,
              background: simRunning ? '#742a2a' : '#276749',
              color: '#fff', fontWeight: 600,
            }}
          >
            {simRunning
              ? <><Square size={12} /> Stop Simulator</>
              : <><Play size={12} /> Start Simulator</>
            }
          </button>
          <button
            onClick={clearAlerts}
            style={{
              padding: '6px 16px', borderRadius: 8,
              border: '1px solid #2d3748', cursor: 'pointer',
              fontSize: 13, background: 'none', color: '#718096',
            }}
          >
            Clear
          </button>
        </div>
      </div>

      <div style={S.main}>
        {/* Stat cards */}
        <div style={S.grid}>
          {[
            {
              title: 'Network IDS F1',
              value: '0.994',
              sub: 'NSL-KDD + CICIDS2017',
              color: '#68d391',
            },
            {
              title: 'False Positive Rate',
              value: '0.29%',
              sub: 'Below 8% target ✓',
              color: '#63b3ed',
            },
            {
              title: 'Total Alerts',
              value: stats.total || 0,
              sub: `${stats.critical || 0} critical · ${stats.high || 0} high`,
              color: '#fc8181',
            },
            {
              title: 'Privacy Budget ε',
              value: '0.119',
              sub: 'Budget: 0.5 — SAFE',
              color: '#f6ad55',
            },
          ].map((c, i) => (
            <div key={i} style={S.card}>
              <div style={S.cardTitle}>{c.title}</div>
              <div style={{ ...S.cardValue, color: c.color }}>
                {c.value}
              </div>
              <div style={S.cardSub}>{c.sub}</div>
            </div>
          ))}
        </div>

        {/* Tabs */}
        <div style={{
          display: 'flex', gap: 4, marginBottom: 20,
          borderBottom: '1px solid #2d3748',
        }}>
          {tabs.map(tab => (
            <button key={tab} onClick={() => setActiveTab(tab)} style={{
              background: 'none', border: 'none', fontSize: 14,
              fontWeight: activeTab === tab ? 600 : 400,
              color: activeTab === tab ? '#63b3ed' : '#718096',
              padding: '8px 20px', cursor: 'pointer',
              borderBottom: activeTab === tab
                ? '2px solid #63b3ed' : '2px solid transparent',
              marginBottom: -1, textTransform: 'capitalize',
            }}>
              {tab === 'alerts' ? `Alerts (${alerts.length})` : tab}
            </button>
          ))}
        </div>

        {/* Alerts tab */}
        {activeTab === 'alerts' && (
          <div style={S.card}>
            <div style={S.sectionTitle}>
              <AlertTriangle size={16} color="#fc8181" />
              Live Alert Feed
              <span style={{
                marginLeft: 'auto', fontSize: 12,
                color: '#718096', fontWeight: 400,
              }}>
                {simRunning
                  ? '● Simulator running'
                  : 'Start simulator for live data'}
              </span>
            </div>
            {alerts.length === 0
              ? <div style={{
                  color: '#718096', fontSize: 14,
                  textAlign: 'center', padding: '40px 0',
                }}>
                  No alerts yet — click Start Simulator
                </div>
              : alerts.map((a, i) => (
                  <AlertCard key={`${a.alert_id}-${i}`} alert={a} />
                ))
            }
          </div>
        )}

        {/* Models tab */}
        {activeTab === 'models' && (
          <div style={S.twoCol}>
            <div style={S.card}>
              <div style={S.sectionTitle}>
                <Activity size={16} color="#63b3ed" />
                Model Comparison
              </div>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart
                  data={metrics.models}
                  margin={{ top: 5, right: 10, left: -20, bottom: 50 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                  <XAxis dataKey="model"
                    tick={{ fill: '#718096', fontSize: 10 }}
                    angle={-30} textAnchor="end"
                  />
                  <YAxis tick={{ fill: '#718096', fontSize: 10 }}
                    domain={[0, 1]}
                  />
                  <Tooltip contentStyle={{
                    background: '#1a1d27',
                    border: '1px solid #2d3748',
                    borderRadius: 8, color: '#e2e8f0',
                  }} />
                  <Bar dataKey="f1"  fill="#63b3ed" name="F1"
                    radius={[4,4,0,0]} />
                  <Bar dataKey="auc" fill="#68d391" name="AUC"
                    radius={[4,4,0,0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div style={S.card}>
              <div style={S.sectionTitle}>
                <Activity size={16} color="#a78bfa" />
                Detailed Metrics
              </div>
              <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                <thead>
                  <tr>
                    {['Model','F1','AUC','FPR%'].map(h => (
                      <th key={h} style={{
                        textAlign: 'left', fontSize: 11,
                        color: '#718096', padding: '6px 8px',
                        borderBottom: '1px solid #2d3748',
                      }}>{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {metrics.models.map((m, i) => (
                    <tr key={i}>
                      <td style={{
                        fontSize: 12, color: '#e2e8f0',
                        padding: '8px', borderBottom: '1px solid #1a1d27',
                      }}>{m.model}</td>
                      <td style={{
                        fontSize: 12, padding: '8px',
                        fontWeight: m.f1 > 0.9 ? 700 : 400,
                        color: m.f1 > 0.9 ? '#68d391' : '#e2e8f0',
                        borderBottom: '1px solid #1a1d27',
                      }}>{m.f1.toFixed(4)}</td>
                      <td style={{
                        fontSize: 12, color: '#e2e8f0',
                        padding: '8px', borderBottom: '1px solid #1a1d27',
                      }}>{m.auc.toFixed(4)}</td>
                      <td style={{
                        fontSize: 12, padding: '8px',
                        color: m.fpr < 0.05 ? '#68d391' : '#f6ad55',
                        borderBottom: '1px solid #1a1d27',
                      }}>{(m.fpr * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Federated tab */}
        {activeTab === 'federated' && (
          <div style={S.twoCol}>
            <div style={S.card}>
              <div style={S.sectionTitle}>
                <Shield size={16} color="#f6ad55" />
                Federated Learning Results
              </div>
              <div style={{
                display: 'grid',
                gridTemplateColumns: '1fr 1fr 1fr',
                gap: 12, marginBottom: 16,
              }}>
                {FEDERATED_ITEMS.map((item, i) => (
                  <div key={i} style={{
                    background: '#0f1117', borderRadius: 8,
                    padding: '10px 12px', textAlign: 'center',
                  }}>
                    <div style={{ fontSize: 11, color: '#718096' }}>
                      {item.label}
                    </div>
                    <div style={{
                      fontSize: 20, fontWeight: 700,
                      color: item.color, marginTop: 4,
                    }}>
                      {item.value}
                    </div>
                  </div>
                ))}
              </div>
              <div style={{
                fontSize: 12, color: '#68d391', textAlign: 'center',
                padding: 8, background: '#1c3229', borderRadius: 6,
              }}>
                ✓ Within 3% target — Privacy budget WITHIN BUDGET
              </div>
            </div>
            <div style={S.card}>
              <div style={S.sectionTitle}>
                <Shield size={16} color="#a78bfa" />
                Privacy Analysis
              </div>
              {[
                ['Mechanism', 'Gaussian Differential Privacy'],
                ['Noise multiplier', '1.1'],
                ['Max gradient norm', '1.0'],
                ['Sigma (σ)', '9.6896'],
                ['Delta (δ)', '1e-5'],
                ['Rounds to converge', '2'],
                ['Fed vs Central gap', '1.96% ✓'],
              ].map(([k, v], i) => (
                <div key={i} style={{
                  display: 'flex', justifyContent: 'space-between',
                  padding: '8px 0',
                  borderBottom: i < 6 ? '1px solid #2d3748' : 'none',
                  fontSize: 13,
                }}>
                  <span style={{ color: '#718096' }}>{k}</span>
                  <span style={{ color: '#fff', fontWeight: 600 }}>{v}</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Ablation tab */}
        {activeTab === 'ablation' && (
          <div style={S.twoCol}>
            <div style={S.card}>
              <div style={S.sectionTitle}>
                <Eye size={16} color="#f6ad55" />
                Ablation Study
              </div>
              <ResponsiveContainer width="100%" height={240}>
                <LineChart data={ABLATION_DATA}
                  margin={{ top: 5, right: 10, left: -20, bottom: 40 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke="#2d3748" />
                  <XAxis dataKey="name"
                    tick={{ fill: '#718096', fontSize: 10 }}
                    angle={-25} textAnchor="end"
                  />
                  <YAxis tick={{ fill: '#718096', fontSize: 10 }}
                    domain={[0.75, 0.85]}
                  />
                  <Tooltip contentStyle={{
                    background: '#1a1d27',
                    border: '1px solid #2d3748',
                    borderRadius: 8, color: '#e2e8f0',
                  }} />
                  <Line type="monotone" dataKey="f1"
                    stroke="#f6ad55" strokeWidth={2}
                    dot={{ fill: '#f6ad55', r: 4 }} name="F1 Score"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div style={S.card}>
              <div style={S.sectionTitle}>
                <Eye size={16} color="#68d391" />
                Novel Contribution Impact
              </div>
              {[
                {
                  label: 'NC1 — Peer Features Added',
                  value: '+4.10% F1', color: '#68d391',
                  desc: 'Role-based peer z-scores vs baseline',
                },
                {
                  label: 'NC1 — Removed from Full',
                  value: '-2.78% F1', color: '#fc8181',
                  desc: 'Drop when peer features excluded',
                },
                {
                  label: 'NC4 — GenAI Features Removed',
                  value: '-1.79% F1', color: '#f6ad55',
                  desc: 'Drop when GenAI signals excluded',
                },
                {
                  label: 'NC3 — Fed vs Central Gap',
                  value: '1.96%', color: '#68d391',
                  desc: 'Within 3% target with ε=0.119',
                },
              ].map((item, i) => (
                <div key={i} style={{
                  padding: '12px 0',
                  borderBottom: i < 3 ? '1px solid #2d3748' : 'none',
                }}>
                  <div style={{
                    display: 'flex', justifyContent: 'space-between',
                  }}>
                    <span style={{ fontSize: 13, color: '#e2e8f0' }}>
                      {item.label}
                    </span>
                    <span style={{
                      fontSize: 15, fontWeight: 700, color: item.color,
                    }}>
                      {item.value}
                    </span>
                  </div>
                  <div style={{ fontSize: 11, color: '#718096', marginTop: 2 }}>
                    {item.desc}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}