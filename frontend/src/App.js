import React, { useEffect, useState, useCallback } from "react";
import axios from "axios";
import {
  Bar, Line, Scatter, Doughnut
} from "react-chartjs-2";
import {
  Chart as ChartJS,
  CategoryScale, LinearScale, BarElement, PointElement,
  LineElement, ArcElement, Title, Tooltip, Legend, Filler
} from "chart.js";

ChartJS.register(
  CategoryScale, LinearScale, BarElement, PointElement,
  LineElement, ArcElement, Title, Tooltip, Legend, Filler
);

const API = process.env.REACT_APP_API || "http://localhost:8000";

// ── Design tokens ────────────────────────────────────────────
const C = {
  bg:       "#0d1117",
  surface:  "#161b22",
  border:   "#30363d",
  text:     "#e6edf3",
  muted:    "#8b949e",
  blue:     "#00d4ff",
  red:      "#ff4757",
  orange:   "#ffa502",
  green:    "#2ed573",
  purple:   "#a29bfe",
};

const chartDefaults = {
  plugins: { legend: { labels: { color: C.text, font: { size: 11 } } } },
  scales: {
    x: { ticks: { color: C.muted, font: { size: 10 } }, grid: { color: C.border } },
    y: { ticks: { color: C.muted, font: { size: 10 } }, grid: { color: C.border } },
  },
  responsive: true,
  maintainAspectRatio: false,
};

// ── Reusable card ────────────────────────────────────────────
const Card = ({ children, style = {} }) => (
  <div style={{
    background: C.surface, border: `1px solid ${C.border}`,
    borderRadius: 10, padding: "18px 20px", ...style
  }}>
    {children}
  </div>
);

const SectionTitle = ({ icon, title, subtitle }) => (
  <div style={{ marginBottom: 14 }}>
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <span style={{ fontSize: 18 }}>{icon}</span>
      <span style={{ color: C.text, fontWeight: 700, fontSize: 15 }}>{title}</span>
    </div>
    {subtitle && <div style={{ color: C.muted, fontSize: 11, marginTop: 2, marginLeft: 26 }}>{subtitle}</div>}
  </div>
);

const Stat = ({ label, value, unit = "", color = C.blue }) => (
  <div style={{ textAlign: "center" }}>
    <div style={{ color, fontSize: 22, fontWeight: 800, fontFamily: "monospace" }}>
      {value}<span style={{ fontSize: 12, color: C.muted, marginLeft: 3 }}>{unit}</span>
    </div>
    <div style={{ color: C.muted, fontSize: 10, marginTop: 2 }}>{label}</div>
  </div>
);

const Badge = ({ text, color }) => (
  <span style={{
    background: color + "22", color, border: `1px solid ${color}44`,
    borderRadius: 4, padding: "2px 8px", fontSize: 10, fontWeight: 600,
    letterSpacing: 0.5, textTransform: "uppercase"
  }}>{text}</span>
);

// ── Main App ─────────────────────────────────────────────────
export default function App() {
  const [stats,      setStats]      = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [anomalies,  setAnomalies]  = useState(null);
  const [insights,   setInsights]   = useState(null);
  const [tinyml,     setTinyml]     = useState(null);
  const [sparkStats, setSparkStats] = useState(null);
  const [pcaInfo,    setPcaInfo]    = useState(null);
  const [modelInfo,  setModelInfo]  = useState(null);
  const [rImage,     setRImage]     = useState(null);

  const [rooms, setRooms]       = useState({ A: "", B: "", C: "" });
  const [roomResults, setRoomResults] = useState([]);
  const [roomChart,   setRoomChart]   = useState(null);
  const [customPower, setCustomPower] = useState("");
  const [analyzeResult, setAnalyzeResult] = useState(null);
  const [loading,     setLoading]   = useState(true);
  const [activeTab,   setActiveTab] = useState("dashboard");

  const fetchAll = useCallback(async () => {
    setLoading(true);
    try {
      const [s, p, a, i, t, sp, pca, mi] = await Promise.all([
        axios.get(`${API}/stats`),
        axios.get(`${API}/prediction`),
        axios.get(`${API}/anomalies`),
        axios.get(`${API}/insights`),
        axios.get(`${API}/tinyml`),
        axios.get(`${API}/spark-stats`),
        axios.get(`${API}/pca-info`),
        axios.get(`${API}/model-info`),
      ]);
      setStats(s.data);
      setPrediction(p.data);
      setAnomalies(a.data);
      setInsights(i.data);
      setTinyml(t.data);
      setSparkStats(sp.data);
      setPcaInfo(pca.data);
      setModelInfo(mi.data);
      setRImage(`${API}/static/plot.png?t=${Date.now()}`);
    } catch (e) {
      console.error(e);
    }
    setLoading(false);
  }, []);

  useEffect(() => { fetchAll(); }, [fetchAll]);

  const analyzeRooms = async () => {
    try {
      const res = await axios.post(`${API}/analyze-rooms`, {
        rooms: [
          { name: "Room A", power: parseFloat(rooms.A) || 0 },
          { name: "Room B", power: parseFloat(rooms.B) || 0 },
          { name: "Room C", power: parseFloat(rooms.C) || 0 },
        ]
      });
      setRoomResults(res.data.results);
      setRoomChart({
        labels: res.data.results.map(r => r.room),
        datasets: [{
          label: "Power (kW)",
          data:  res.data.results.map(r => r.power),
          backgroundColor: res.data.results.map(r =>
            r.anomaly ? C.red + "cc" : r.decision.label === "HIGH" ? C.orange + "cc" : C.blue + "cc"
          ),
          borderRadius: 6,
        }]
      });
    } catch (e) { alert("Backend unreachable 🚨"); }
  };

  const analyzeSingle = async () => {
    if (!customPower) return;
    try {
      const res = await axios.get(`${API}/analyze?power=${customPower}`);
      setAnalyzeResult(res.data);
    } catch (e) { alert("Error calling /analyze"); }
  };

  // ── Shared chart data ──────────────────────────────────────
  const predChartData = prediction ? {
    labels: prediction.Global_active_power.map((_, i) => i),
    datasets: [
      {
        label: "Actual Power",
        data: prediction.Global_active_power,
        borderColor: C.blue,
        backgroundColor: C.blue + "18",
        borderWidth: 1.5,
        pointRadius: 0,
        fill: true,
        tension: 0.4,
      },
      {
        label: "Rolling Forecast (24h)",
        data: prediction.prediction,
        borderColor: C.orange,
        borderWidth: 2,
        pointRadius: 0,
        borderDash: [5, 3],
        tension: 0.4,
      }
    ]
  } : null;

  const anomalyDoughnut = stats ? {
    labels: ["Normal", "Anomaly"],
    datasets: [{
      data: [
        stats.total_records - stats.anomaly_count,
        stats.anomaly_count,
      ],
      backgroundColor: [C.green + "bb", C.red + "bb"],
      borderColor: [C.green, C.red],
      borderWidth: 2,
    }]
  } : null;

  const pcaBarData = pcaInfo ? {
    labels: pcaInfo.explained_variance.map((_, i) => `PC${i + 1}`),
    datasets: [{
      label: "Explained Variance (%)",
      data: pcaInfo.explained_variance,
      backgroundColor: [C.blue, C.purple, C.orange, C.green, C.red].map(c => c + "cc"),
      borderRadius: 5,
    }]
  } : null;

  // ── Decision color ─────────────────────────────────────────
  const decisionColor = (label) =>
    label === "CRITICAL" ? C.red :
    label === "HIGH"     ? C.orange :
    label === "LOW"      ? C.purple :
    C.green;

  const insightColor = (type) =>
    type === "critical" ? C.red :
    type === "warning"  ? C.orange :
    type === "alert"    ? C.orange :
    type === "model"    ? C.purple :
    C.green;

  // ── TABS ───────────────────────────────────────────────────
  const tabs = [
    { id: "dashboard", label: "📊 Dashboard" },
    { id: "anomaly",   label: "🔍 Anomalies" },
    { id: "rooms",     label: "🏠 Room Analysis" },
    { id: "model",     label: "🧠 Model Info" },
    { id: "r-analysis",label: "📈 R Analysis" },
  ];

  const tabStyle = (id) => ({
    padding: "8px 18px",
    background:   activeTab === id ? C.blue + "22" : "transparent",
    border:       `1px solid ${activeTab === id ? C.blue : C.border}`,
    borderRadius: 6,
    color:        activeTab === id ? C.blue : C.muted,
    cursor: "pointer",
    fontSize: 12,
    fontWeight: activeTab === id ? 700 : 400,
    whiteSpace: "nowrap",
  });

  return (
    <div style={{ background: C.bg, minHeight: "100vh", fontFamily: "'DM Mono', 'Courier New', monospace", color: C.text }}>

      {/* ── HEADER ─────────────────────────────────────────── */}
      <div style={{
        borderBottom: `1px solid ${C.border}`,
        padding: "16px 28px",
        display: "flex", alignItems: "center", justifyContent: "space-between",
        background: C.surface,
      }}>
        <div>
          <div style={{ fontSize: 18, fontWeight: 800, color: C.blue, letterSpacing: -0.5 }}>
            ⚡ Smart Energy Optimization System
          </div>
          <div style={{ color: C.muted, fontSize: 11, marginTop: 2 }}>
            Isolation Forest · PCA · TinyML · Apache Spark · R Statistics · Docker
          </div>
        </div>
        <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
          <Badge text="POWERCON 2018" color={C.purple} />
          <Badge text="v2.0" color={C.blue} />
          <button onClick={fetchAll}
            style={{ background: C.blue + "22", border: `1px solid ${C.blue}`, color: C.blue,
                     borderRadius: 6, padding: "6px 14px", cursor: "pointer", fontSize: 11 }}>
            ↺ Refresh
          </button>
        </div>
      </div>

      {/* ── TABS ───────────────────────────────────────────── */}
      <div style={{ padding: "12px 28px", display: "flex", gap: 8, borderBottom: `1px solid ${C.border}`, flexWrap: "wrap" }}>
        {tabs.map(t => (
          <button key={t.id} style={tabStyle(t.id)} onClick={() => setActiveTab(t.id)}>
            {t.label}
          </button>
        ))}
      </div>

      {loading && (
        <div style={{ textAlign: "center", padding: 60, color: C.muted }}>
          <div style={{ fontSize: 32, marginBottom: 12 }}>⚡</div>
          Loading data from backend…
        </div>
      )}

      <div style={{ padding: "24px 28px", maxWidth: 1400, margin: "0 auto" }}>

        {/* ════════════════════════════════════════════════════
            TAB: DASHBOARD
        ════════════════════════════════════════════════════ */}
        {activeTab === "dashboard" && !loading && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

            {/* KPI Row */}
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 14 }}>
              {stats && <>
                <Card><Stat label="Mean Power"   value={stats.mean_power.toFixed(3)} unit="kW" color={C.blue} /></Card>
                <Card><Stat label="Peak Power"   value={stats.max_power.toFixed(3)}  unit="kW" color={C.red} /></Card>
                <Card><Stat label="Min Power"    value={stats.min_power.toFixed(3)}  unit="kW" color={C.green} /></Card>
                <Card><Stat label="Std Deviation" value={stats.std_power.toFixed(3)} unit="kW" color={C.orange} /></Card>
                <Card><Stat label="Anomaly Rate" value={stats.anomaly_rate.toFixed(1)} unit="%" color={C.red} /></Card>
                <Card><Stat label="Total Records" value={stats.total_records.toLocaleString()} color={C.purple} /></Card>
              </>}
            </div>

            {/* TinyML banner */}
            {tinyml && (
              <Card style={{ borderColor: decisionColor(tinyml.decision.label), borderWidth: 2 }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: 10 }}>
                  <div>
                    <SectionTitle icon="🤖" title="TinyML Real-time Decision" subtitle="Simulated MCU inference engine (no hardware required)" />
                    <div style={{ display: "flex", gap: 24, marginLeft: 26 }}>
                      <Stat label="Latest Power" value={tinyml.power.toFixed(4)} unit="kW" color={C.text} />
                      <Stat label="Decision" value={tinyml.decision.emoji} color={decisionColor(tinyml.decision.label)} />
                    </div>
                  </div>
                  <div style={{ textAlign: "right" }}>
                    <Badge text={tinyml.decision.label} color={decisionColor(tinyml.decision.label)} />
                    <div style={{ color: C.muted, fontSize: 12, marginTop: 8 }}>{tinyml.decision.action}</div>
                    <div style={{ color: C.muted, fontSize: 11, marginTop: 4 }}>
                      Confidence: <span style={{ color: C.orange }}>{tinyml.decision.confidence}%</span>
                    </div>
                  </div>
                </div>
              </Card>
            )}

            {/* Prediction Chart */}
            {predChartData && (
              <Card>
                <SectionTitle icon="📉" title="Power Consumption + Rolling 24h Forecast"
                  subtitle="Rolling mean baseline (forecasting.py) — foundation for future LSTM integration" />
                <div style={{ height: 280 }}>
                  <Line data={predChartData} options={{
                    ...chartDefaults,
                    plugins: { ...chartDefaults.plugins,
                      title: { display: false }
                    }
                  }} />
                </div>
                {prediction?.rmse !== undefined && (
                  <div style={{ color: C.muted, fontSize: 11, marginTop: 8, textAlign: "right" }}>
                    RMSE: <span style={{ color: C.orange }}>{prediction.rmse}</span> kW
                  </div>
                )}
              </Card>
            )}

            {/* Insights + Doughnut */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 320px", gap: 16 }}>
              {insights && (
                <Card>
                  <SectionTitle icon="💡" title="AI Insights" subtitle="Automated pattern detection based on paper §IV" />
                  {insights.insights.map((ins, i) => (
                    <div key={i} style={{
                      padding: "10px 14px", marginBottom: 8,
                      background: insightColor(ins.type) + "11",
                      borderLeft: `3px solid ${insightColor(ins.type)}`,
                      borderRadius: "0 6px 6px 0",
                      fontSize: 13, color: C.text,
                    }}>
                      {ins.msg}
                    </div>
                  ))}
                  <div style={{ color: C.muted, fontSize: 11, marginTop: 8 }}>
                    Anomaly rate: <span style={{ color: C.red }}>{insights.anomaly_rate}%</span>
                  </div>
                </Card>
              )}
              {anomalyDoughnut && (
                <Card>
                  <SectionTitle icon="🎯" title="Normal vs Anomaly" />
                  <div style={{ height: 200 }}>
                    <Doughnut data={anomalyDoughnut} options={{
                      plugins: { legend: { labels: { color: C.text } } },
                      maintainAspectRatio: false,
                    }} />
                  </div>
                  {stats && (
                    <div style={{ marginTop: 10, display: "flex", justifyContent: "space-around" }}>
                      <Stat label="Normal" value={stats.total_records - stats.anomaly_count} color={C.green} />
                      <Stat label="Anomaly" value={stats.anomaly_count} color={C.red} />
                    </div>
                  )}
                </Card>
              )}
            </div>

            {/* Spark Stats */}
            {sparkStats && (
              <Card>
                <SectionTitle icon="🔥" title="Apache Spark Processing Results"
                  subtitle="Data processed via PySpark — partitioned & aggregated at scale (paper §IV-B)" />
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 12 }}>
                  <Stat label="Total Rows" value={sparkStats.rows.toLocaleString()} color={C.orange} />
                  <Stat label="Avg Power"  value={sparkStats.avg_power?.toFixed(4)} unit="kW" color={C.blue} />
                  <Stat label="Max Power"  value={sparkStats.max_power?.toFixed(4)} unit="kW" color={C.red} />
                  <Stat label="Min Power"  value={sparkStats.min_power?.toFixed(4)} unit="kW" color={C.green} />
                  <Stat label="Std Power"  value={sparkStats.std_power?.toFixed(4)} unit="kW" color={C.purple} />
                </div>
                <div style={{ marginTop: 10, color: C.muted, fontSize: 10 }}>
                  Source: {sparkStats.source}
                </div>
              </Card>
            )}
          </div>
        )}

        {/* ════════════════════════════════════════════════════
            TAB: ANOMALIES
        ════════════════════════════════════════════════════ */}
        {activeTab === "anomaly" && !loading && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>

            {/* Single-point analyzer */}
            <Card>
              <SectionTitle icon="🔬" title="Single-Point Anomaly Analyzer"
                subtitle="Enter a power reading → get IForest z-score + TinyML decision (paper §IV, /analyze endpoint)" />
              <div style={{ display: "flex", gap: 10, alignItems: "flex-end", flexWrap: "wrap" }}>
                <div>
                  <div style={{ color: C.muted, fontSize: 11, marginBottom: 4 }}>Power Value (kW)</div>
                  <input
                    type="number" step="0.1" min="0" placeholder="e.g. 3.5"
                    value={customPower}
                    onChange={e => setCustomPower(e.target.value)}
                    style={{
                      background: C.bg, border: `1px solid ${C.border}`,
                      borderRadius: 6, padding: "8px 14px", color: C.text,
                      fontSize: 14, width: 160,
                    }}
                  />
                </div>
                <button onClick={analyzeSingle} style={{
                  background: C.blue, border: "none", borderRadius: 6,
                  padding: "9px 20px", color: "#000", fontWeight: 700,
                  fontSize: 13, cursor: "pointer",
                }}>
                  Analyze →
                </button>
              </div>
              {analyzeResult && (
                <div style={{
                  marginTop: 16, padding: "14px 18px",
                  background: analyzeResult.anomaly ? C.red + "11" : C.green + "11",
                  border: `1px solid ${analyzeResult.anomaly ? C.red : C.green}`,
                  borderRadius: 8, display: "flex", gap: 30, flexWrap: "wrap",
                }}>
                  <Stat label="Power"      value={analyzeResult.power} unit="kW" color={C.text} />
                  <Stat label="Z-Score"    value={analyzeResult.z_score} color={analyzeResult.anomaly ? C.red : C.green} />
                  <Stat label="Decision"   value={analyzeResult.decision?.emoji + " " + analyzeResult.decision?.label}
                        color={decisionColor(analyzeResult.decision?.label)} />
                  <div>
                    <div style={{ color: C.muted, fontSize: 10 }}>Anomaly</div>
                    <Badge text={analyzeResult.anomaly ? "YES 🚨" : "NO ✅"}
                           color={analyzeResult.anomaly ? C.red : C.green} />
                  </div>
                  <div style={{ flex: 1 }}>
                    <div style={{ color: C.muted, fontSize: 10 }}>Remedy</div>
                    <div style={{ color: C.text, fontSize: 13, marginTop: 3 }}>💡 {analyzeResult.remedy}</div>
                  </div>
                </div>
              )}
            </Card>

            {/* Detected anomalies table */}
            {anomalies && (
              <Card>
                <SectionTitle icon="⚠️" title={`Detected Anomalies (last ${anomalies.Global_active_power.length})`}
                  subtitle="Flagged by Isolation Forest with contamination=0.01, 100 iTrees, 256 samples (paper §II)" />
                <div style={{ overflowX: "auto" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                    <thead>
                      <tr style={{ borderBottom: `1px solid ${C.border}` }}>
                        {["#", "Power (kW)", "Score", "Severity"].map(h => (
                          <th key={h} style={{ padding: "8px 12px", textAlign: "left", color: C.muted }}>{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {anomalies.Global_active_power.slice(0, 15).map((p, i) => {
                        const sc = anomalies.anomaly_scores?.[i] ?? 0;
                        const sev = p > 4 ? "CRITICAL" : p > 2 ? "HIGH" : "MODERATE";
                        return (
                          <tr key={i} style={{ borderBottom: `1px solid ${C.border}22` }}>
                            <td style={{ padding: "7px 12px", color: C.muted }}>{i + 1}</td>
                            <td style={{ padding: "7px 12px", color: C.red, fontFamily: "monospace" }}>
                              {p.toFixed(4)}
                            </td>
                            <td style={{ padding: "7px 12px", color: C.orange, fontFamily: "monospace" }}>
                              {sc.toFixed(4)}
                            </td>
                            <td style={{ padding: "7px 12px" }}>
                              <Badge text={sev} color={sev === "CRITICAL" ? C.red : sev === "HIGH" ? C.orange : C.purple} />
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <div style={{ color: C.muted, fontSize: 10, marginTop: 10 }}>
                  Total anomalies: <span style={{ color: C.red }}>{stats?.anomaly_count}</span> &nbsp;|&nbsp;
                  Rate: <span style={{ color: C.red }}>{stats?.anomaly_rate}%</span> &nbsp;|&nbsp;
                  Paper threshold: contamination = 0.01 (1%)
                </div>
              </Card>
            )}
          </div>
        )}

        {/* ════════════════════════════════════════════════════
            TAB: ROOM ANALYSIS
        ════════════════════════════════════════════════════ */}
        {activeTab === "rooms" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <Card>
              <SectionTitle icon="🏠" title="Multi-Room Energy Analysis"
                subtitle="Enter kW readings per room → TinyML decision + z-score anomaly detection" />
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12, maxWidth: 500 }}>
                {["A", "B", "C"].map(r => (
                  <div key={r}>
                    <div style={{ color: C.muted, fontSize: 11, marginBottom: 4 }}>Room {r} (kW)</div>
                    <input
                      type="number" step="0.1" min="0" placeholder="0.0"
                      value={rooms[r]}
                      onChange={e => setRooms({ ...rooms, [r]: e.target.value })}
                      style={{
                        width: "100%", boxSizing: "border-box",
                        background: C.bg, border: `1px solid ${C.border}`,
                        borderRadius: 6, padding: "8px 12px",
                        color: C.text, fontSize: 14,
                      }}
                    />
                  </div>
                ))}
              </div>
              <button onClick={analyzeRooms} style={{
                marginTop: 14, background: C.blue, border: "none",
                borderRadius: 6, padding: "9px 22px",
                color: "#000", fontWeight: 700, fontSize: 13, cursor: "pointer",
              }}>
                Analyze Rooms →
              </button>
            </Card>

            {roomResults.length > 0 && (
              <>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 14 }}>
                  {roomResults.map((r, i) => (
                    <Card key={i} style={{
                      borderColor: r.anomaly ? C.red : decisionColor(r.decision.label),
                      borderWidth: 2,
                    }}>
                      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 10 }}>
                        <div style={{ fontWeight: 700, fontSize: 15 }}>{r.room}</div>
                        <Badge text={r.decision.label} color={decisionColor(r.decision.label)} />
                      </div>
                      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 10 }}>
                        <Stat label="Power" value={r.power} unit="kW" color={C.text} />
                        <Stat label="Z-Score" value={r.z_score} color={r.anomaly ? C.red : C.green} />
                      </div>
                      <div style={{ fontSize: 20, marginBottom: 6 }}>{r.decision.emoji} {r.decision.action}</div>
                      <div style={{ color: C.muted, fontSize: 12 }}>💡 {r.remedy}</div>
                      {r.anomaly && (
                        <div style={{
                          marginTop: 10, padding: "6px 10px",
                          background: C.red + "18", border: `1px solid ${C.red}44`,
                          borderRadius: 5, fontSize: 12, color: C.red,
                        }}>
                          🚨 Anomaly detected (z-score &gt; 2σ)
                        </div>
                      )}
                    </Card>
                  ))}
                </div>

                {roomChart && (
                  <Card>
                    <SectionTitle icon="📊" title="Room Power Comparison" />
                    <div style={{ height: 220 }}>
                      <Bar data={roomChart} options={chartDefaults} />
                    </div>
                  </Card>
                )}
              </>
            )}
          </div>
        )}

        {/* ════════════════════════════════════════════════════
            TAB: MODEL INFO
        ════════════════════════════════════════════════════ */}
        {activeTab === "model" && !loading && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16 }}>

              {modelInfo && (
                <Card>
                  <SectionTitle icon="🌲" title="Isolation Forest (iForest)"
                    subtitle="Paper §II — Liu, Ting, Zhou. ICDM 2008" />
                  {[
                    ["Algorithm",       modelInfo.algorithm],
                    ["n_estimators",    `${modelInfo.n_estimators} trees (paper: path well-covered at 100)`],
                    ["max_samples",     `${modelInfo.max_samples} (paper: avg 256 samples)`],
                    ["contamination",   `${modelInfo.contamination} (paper: anomalies are 'small fraction')`],
                    ["Type",            modelInfo.type],
                    ["Feature Reduction", modelInfo.feature_reduction],
                    ["Anomalies found", `${modelInfo.anomaly_count} / ${modelInfo.total_records} (${modelInfo.anomaly_rate_pct}%)`],
                    ["Paper accuracy",  modelInfo.paper_accuracy],
                  ].map(([k, v]) => (
                    <div key={k} style={{
                      display: "flex", justifyContent: "space-between",
                      padding: "7px 0", borderBottom: `1px solid ${C.border}22`,
                      fontSize: 12, flexWrap: "wrap", gap: 4,
                    }}>
                      <span style={{ color: C.muted }}>{k}</span>
                      <span style={{ color: C.text, textAlign: "right", maxWidth: "60%" }}>{v}</span>
                    </div>
                  ))}
                </Card>
              )}

              {pcaInfo && (
                <Card>
                  <SectionTitle icon="🔬" title="PCA Dimensionality Reduction"
                    subtitle="Paper §III-B — reduces 13 features → 5 principal components" />
                  {[
                    ["Method",         pcaInfo.method],
                    ["n_components",   pcaInfo.n_components],
                    ["Original features", pcaInfo.original_features],
                    ["Total variance", `${pcaInfo.total_variance_explained}%`],
                  ].map(([k, v]) => (
                    <div key={k} style={{
                      display: "flex", justifyContent: "space-between",
                      padding: "7px 0", borderBottom: `1px solid ${C.border}22`, fontSize: 12,
                    }}>
                      <span style={{ color: C.muted }}>{k}</span>
                      <span style={{ color: C.text }}>{v}</span>
                    </div>
                  ))}
                  <div style={{ marginTop: 14 }}>
                    <div style={{ color: C.muted, fontSize: 11, marginBottom: 8 }}>Variance per component:</div>
                    {pcaInfo.explained_variance.map((v, i) => (
                      <div key={i} style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 5 }}>
                        <span style={{ color: C.muted, fontSize: 11, width: 28 }}>PC{i+1}</span>
                        <div style={{ flex: 1, background: C.border, borderRadius: 3, height: 8 }}>
                          <div style={{ width: `${v}%`, background: C.blue, borderRadius: 3, height: "100%" }} />
                        </div>
                        <span style={{ color: C.blue, fontSize: 11, width: 40 }}>{v}%</span>
                      </div>
                    ))}
                  </div>
                  {pcaBarData && (
                    <div style={{ height: 160, marginTop: 14 }}>
                      <Bar data={pcaBarData} options={{
                        ...chartDefaults,
                        plugins: { legend: { display: false } },
                      }} />
                    </div>
                  )}
                  <div style={{ marginTop: 12, color: C.muted, fontSize: 11 }}>
                    📌 {pcaInfo.note}
                  </div>
                </Card>
              )}
            </div>

            {/* Feature Engineering explanation */}
            <Card>
              <SectionTitle icon="⚙️" title="Feature Engineering Pipeline (Paper §III-A)"
                subtitle="13 features extracted per 24-sample window, then PCA-reduced to 5 components" />
              <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 12 }}>
                {[
                  { name: "Mean-based Features", count: 7, desc: "Average consumption per weekday bucket (Mon–Sun). Paper §III-A-2.", color: C.blue },
                  { name: "Statistical Features", count: 5, desc: "mean, std, min, max, range per window. Baseline descriptors.", color: C.orange },
                  { name: "Trend Features (D, R)", count: 2, desc: "Sliding window downward (D) and rising (R) indices. Paper §III-A-3. Key for theft detection.", color: C.red },
                ].map((f, i) => (
                  <div key={i} style={{
                    padding: "14px 16px",
                    background: f.color + "11",
                    border: `1px solid ${f.color}44`,
                    borderRadius: 8,
                  }}>
                    <div style={{ color: f.color, fontWeight: 700, fontSize: 13, marginBottom: 4 }}>
                      {f.name}
                    </div>
                    <div style={{ color: C.text, fontSize: 22, fontWeight: 800, marginBottom: 6 }}>{f.count}</div>
                    <div style={{ color: C.muted, fontSize: 11, lineHeight: 1.5 }}>{f.desc}</div>
                  </div>
                ))}
              </div>
            </Card>

            {/* TinyML thresholds */}
            {tinyml && (
              <Card>
                <SectionTitle icon="🤖" title="TinyML Decision Thresholds"
                  subtitle="Software-only inference engine — simulates MCU rule-based classifier" />
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 10 }}>
                  {[
                    { label: "LOW",      range: `< ${tinyml.thresholds.low} kW`,     color: C.purple, desc: "Standby / idle" },
                    { label: "NORMAL",   range: `${tinyml.thresholds.low}–${tinyml.thresholds.high} kW`, color: C.green, desc: "Optimal usage" },
                    { label: "HIGH",     range: `${tinyml.thresholds.high}–${tinyml.thresholds.warning} kW`, color: C.orange, desc: "Reduce load" },
                    { label: "CRITICAL", range: `> ${tinyml.thresholds.warning} kW`, color: C.red, desc: "Immediate action" },
                  ].map(t => (
                    <div key={t.label} style={{
                      padding: "12px 14px",
                      background: t.color + "11",
                      border: `1px solid ${t.color}44`,
                      borderRadius: 8, textAlign: "center",
                    }}>
                      <Badge text={t.label} color={t.color} />
                      <div style={{ color: t.color, fontSize: 13, fontFamily: "monospace", margin: "8px 0 4px" }}>{t.range}</div>
                      <div style={{ color: C.muted, fontSize: 11 }}>{t.desc}</div>
                    </div>
                  ))}
                </div>
              </Card>
            )}
          </div>
        )}

        {/* ════════════════════════════════════════════════════
            TAB: R ANALYSIS
        ════════════════════════════════════════════════════ */}
        {activeTab === "r-analysis" && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            <Card>
              <SectionTitle icon="📈" title="R Statistical Analysis Output"
                subtitle="Multi-panel plot generated by analysis.R — distribution, time-series, boxplot, ROC-style evaluation" />
              {rImage ? (
                <div>
                  <img src={rImage} alt="R Analysis Plot"
                    style={{ width: "100%", borderRadius: 8, border: `1px solid ${C.border}` }}
                    onError={() => setRImage(null)}
                  />
                  <div style={{ color: C.muted, fontSize: 11, marginTop: 8, textAlign: "center" }}>
                    Generated by R (r-base:4.3.1 Docker container) · analysis.R
                  </div>
                </div>
              ) : (
                <div style={{
                  height: 200, display: "flex", alignItems: "center", justifyContent: "center",
                  border: `2px dashed ${C.border}`, borderRadius: 8, color: C.muted, flexDirection: "column", gap: 8,
                }}>
                  <div style={{ fontSize: 32 }}>📊</div>
                  <div>R plot not yet generated</div>
                  <div style={{ fontSize: 11 }}>Run: docker-compose up r-analysis</div>
                </div>
              )}
            </Card>

            {/* R code snippet */}
            <Card>
              <SectionTitle icon="💻" title="R Script Summary (analysis.R)"
                subtitle="Runs inside Docker — no local R installation required" />
              <div style={{
                background: C.bg, borderRadius: 6, padding: "14px 16px",
                fontFamily: "monospace", fontSize: 11, lineHeight: 1.7,
                overflowX: "auto", border: `1px solid ${C.border}`,
              }}>
                {[
                  <><span style={{ color: C.green }}>## Paper §III-A-1: Load & clean</span></>,
                  <><span style={{ color: C.blue }}>data</span> {"<-"} <span style={{ color: C.orange }}>read.csv</span>(<span style={{ color: "#a8ff78" }}>"processed_spark_data.csv"</span>)</>,
                  <><span style={{ color: C.blue }}>data</span> {"<-"} data[data$Global_active_power {">"} 0, ]</>,
                  "",
                  <><span style={{ color: C.green }}>## Anomaly threshold: μ + 2σ (Paper §III-A)</span></>,
                  <>thr {"<-"} <span style={{ color: C.orange }}>mean</span>(gap) + 2 * <span style={{ color: C.orange }}>sd</span>(gap)</>,
                  "",
                  <><span style={{ color: C.green }}>## Multi-panel: hist, time-series, boxplot, density, ROC</span></>,
                  <><span style={{ color: C.orange }}>png</span>(<span style={{ color: "#a8ff78" }}>"plot.png"</span>, width=1200, height=900, res=120)</>,
                  <><span style={{ color: C.orange }}>par</span>(mfrow = <span style={{ color: C.orange }}>c</span>(2, 3))</>,
                ].map((line, i) => (
                  <div key={i}>{line || <>&nbsp;</>}</div>
                ))}
              </div>
            </Card>
          </div>
        )}
      </div>

      {/* ── FOOTER ─────────────────────────────────────────── */}
      <div style={{
        borderTop: `1px solid ${C.border}`, padding: "14px 28px",
        color: C.muted, fontSize: 10, display: "flex",
        justifyContent: "space-between", flexWrap: "wrap", gap: 8,
      }}>
        <div>Smart Energy Optimization System · Hardware &amp; Software Workshop</div>
        <div>Based on: Wei Mao et al., POWERCON 2018 · Isolation Forest + PCA + TinyML + Spark + R</div>
      </div>
    </div>
  );
}