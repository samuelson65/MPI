import { useState, useMemo } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Cell, ResponsiveContainer } from "recharts";

// ── Mock historical risk table (replace with your real aggregated data) ──────
const RISK_TABLE = [
  { CPT_Code: "99213", Total_Occurrences: 412, Total_Drops: 187, Drop_Percentage: 45.39 },
  { CPT_Code: "99214", Total_Occurrences: 298, Total_Drops: 119, Drop_Percentage: 39.93 },
  { CPT_Code: "85025",  Total_Occurrences: 201, Total_Drops: 68,  Drop_Percentage: 33.83 },
  { CPT_Code: "93000",  Total_Occurrences: 175, Total_Drops: 52,  Drop_Percentage: 29.71 },
  { CPT_Code: "99215",  Total_Occurrences: 143, Total_Drops: 38,  Drop_Percentage: 26.57 },
  { CPT_Code: "80053",  Total_Occurrences: 310, Total_Drops: 77,  Drop_Percentage: 24.84 },
  { CPT_Code: "71046",  Total_Occurrences: 88,  Total_Drops: 19,  Drop_Percentage: 21.59 },
  { CPT_Code: "36415",  Total_Occurrences: 520, Total_Drops: 104, Drop_Percentage: 20.0  },
  { CPT_Code: "99212",  Total_Occurrences: 267, Total_Drops: 43,  Drop_Percentage: 16.10 },
  { CPT_Code: "97110",  Total_Occurrences: 134, Total_Drops: 18,  Drop_Percentage: 13.43 },
];

const riskLevel = (pct) => {
  if (pct >= 40) return { label: "HIGH", color: "#ef4444", bg: "#fef2f2", border: "#fca5a5" };
  if (pct >= 25) return { label: "MED",  color: "#f59e0b", bg: "#fffbeb", border: "#fcd34d" };
  return               { label: "LOW",  color: "#22c55e", bg: "#f0fdf4", border: "#86efac" };
};

const BAR_COLORS = RISK_TABLE.map(r => riskLevel(r.Drop_Percentage).color);

export default function App() {
  const [input, setInput] = useState("99213, 85025, 99215");
  const [minOcc, setMinOcc] = useState(30);
  const [analyzed, setAnalyzed] = useState(null);

  const filteredRisk = useMemo(
    () => RISK_TABLE.filter(r => r.Total_Occurrences >= minOcc),
    [minOcc]
  );

  const analyze = () => {
    const codes = input.split(",").map(s => s.trim().toUpperCase()).filter(Boolean);
    const matched = filteredRisk
      .filter(r => codes.includes(r.CPT_Code))
      .sort((a, b) => b.Drop_Percentage - a.Drop_Percentage);
    const unknown = codes.filter(c => !filteredRisk.find(r => r.CPT_Code === c));
    setAnalyzed({ matched, unknown, codes });
  };

  const overallRisk = useMemo(() => {
    if (!analyzed?.matched?.length) return null;
    const avg = analyzed.matched.reduce((s, r) => s + r.Drop_Percentage, 0) / analyzed.matched.length;
    return avg;
  }, [analyzed]);

  return (
    <div style={{
      fontFamily: "'DM Mono', 'Courier New', monospace",
      background: "#0f0f13",
      minHeight: "100vh",
      color: "#e2e2e8",
      padding: "32px 24px",
    }}>
      {/* Header */}
      <div style={{ maxWidth: 900, margin: "0 auto" }}>
        <div style={{ display: "flex", alignItems: "baseline", gap: 12, marginBottom: 4 }}>
          <span style={{ fontSize: 11, letterSpacing: 4, color: "#6366f1", textTransform: "uppercase" }}>
            Clinical Claims Intelligence
          </span>
        </div>
        <h1 style={{ margin: "0 0 4px", fontSize: 28, fontWeight: 700, letterSpacing: -1, color: "#fff" }}>
          CPT Denial Risk Analyzer
        </h1>
        <p style={{ margin: "0 0 32px", fontSize: 13, color: "#71717a" }}>
          Score incoming claims against historical denial patterns before submission.
        </p>

        {/* Controls */}
        <div style={{
          background: "#18181f",
          border: "1px solid #27272d",
          borderRadius: 12,
          padding: "20px 24px",
          marginBottom: 24,
        }}>
          <div style={{ display: "flex", gap: 16, alignItems: "flex-end", flexWrap: "wrap" }}>
            <div style={{ flex: 1, minWidth: 200 }}>
              <label style={{ fontSize: 11, color: "#71717a", letterSpacing: 2, display: "block", marginBottom: 6 }}>
                CPT CODES (comma-separated)
              </label>
              <input
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => e.key === "Enter" && analyze()}
                placeholder="e.g. 99213, 85025, 93000"
                style={{
                  width: "100%",
                  background: "#0f0f13",
                  border: "1px solid #3f3f46",
                  borderRadius: 8,
                  padding: "10px 14px",
                  color: "#e2e2e8",
                  fontSize: 14,
                  fontFamily: "inherit",
                  boxSizing: "border-box",
                  outline: "none",
                }}
              />
            </div>
            <div style={{ minWidth: 140 }}>
              <label style={{ fontSize: 11, color: "#71717a", letterSpacing: 2, display: "block", marginBottom: 6 }}>
                MIN OCCURRENCES
              </label>
              <input
                type="number"
                value={minOcc}
                onChange={e => setMinOcc(Number(e.target.value))}
                style={{
                  width: "100%",
                  background: "#0f0f13",
                  border: "1px solid #3f3f46",
                  borderRadius: 8,
                  padding: "10px 14px",
                  color: "#e2e2e8",
                  fontSize: 14,
                  fontFamily: "inherit",
                  boxSizing: "border-box",
                  outline: "none",
                }}
              />
            </div>
            <button
              onClick={analyze}
              style={{
                background: "#6366f1",
                color: "#fff",
                border: "none",
                borderRadius: 8,
                padding: "10px 24px",
                fontSize: 13,
                fontWeight: 600,
                cursor: "pointer",
                letterSpacing: 1,
                fontFamily: "inherit",
                whiteSpace: "nowrap",
              }}
            >
              ANALYZE CLAIM →
            </button>
          </div>
        </div>

        {/* Results */}
        {analyzed && (
          <div style={{ display: "flex", flexDirection: "column", gap: 20 }}>
            {/* Overall risk badge */}
            {overallRisk !== null && (() => {
              const r = riskLevel(overallRisk);
              return (
                <div style={{
                  background: "#18181f",
                  border: `1px solid ${r.border}`,
                  borderRadius: 12,
                  padding: "18px 24px",
                  display: "flex",
                  alignItems: "center",
                  gap: 20,
                }}>
                  <div style={{
                    background: r.color,
                    color: "#fff",
                    borderRadius: 8,
                    padding: "6px 14px",
                    fontSize: 12,
                    fontWeight: 700,
                    letterSpacing: 3,
                  }}>
                    {r.label} RISK
                  </div>
                  <div>
                    <span style={{ fontSize: 28, fontWeight: 700, color: r.color }}>
                      {overallRisk.toFixed(1)}%
                    </span>
                    <span style={{ fontSize: 13, color: "#71717a", marginLeft: 8 }}>
                      avg drop rate across {analyzed.matched.length} matched code{analyzed.matched.length !== 1 ? "s" : ""}
                    </span>
                  </div>
                </div>
              );
            })()}

            {/* Unknown codes warning */}
            {analyzed.unknown.length > 0 && (
              <div style={{
                background: "#1c1a10",
                border: "1px solid #78350f",
                borderRadius: 10,
                padding: "12px 18px",
                fontSize: 13,
                color: "#fbbf24",
              }}>
                ⚠ No historical data for: <strong>{analyzed.unknown.join(", ")}</strong>
                {" "}— may be below the {minOcc}-occurrence threshold or unseen codes.
              </div>
            )}

            {/* Code breakdown table */}
            {analyzed.matched.length > 0 && (
              <div style={{
                background: "#18181f",
                border: "1px solid #27272d",
                borderRadius: 12,
                overflow: "hidden",
              }}>
                <div style={{ padding: "14px 20px", borderBottom: "1px solid #27272d" }}>
                  <span style={{ fontSize: 11, color: "#71717a", letterSpacing: 3, textTransform: "uppercase" }}>
                    Code-Level Breakdown
                  </span>
                </div>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                  <thead>
                    <tr style={{ borderBottom: "1px solid #27272d" }}>
                      {["CPT Code", "Occurrences", "Drops", "Drop %", "Risk"].map(h => (
                        <th key={h} style={{
                          padding: "10px 20px",
                          fontSize: 11,
                          color: "#52525b",
                          letterSpacing: 2,
                          textAlign: "left",
                          fontWeight: 600,
                        }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {analyzed.matched.map((r, i) => {
                      const risk = riskLevel(r.Drop_Percentage);
                      return (
                        <tr key={r.CPT_Code} style={{
                          borderBottom: i < analyzed.matched.length - 1 ? "1px solid #1f1f26" : "none",
                          background: i % 2 === 0 ? "transparent" : "#14141a",
                        }}>
                          <td style={{ padding: "12px 20px", fontWeight: 700, color: "#a5b4fc", fontSize: 15 }}>
                            {r.CPT_Code}
                          </td>
                          <td style={{ padding: "12px 20px", color: "#a1a1aa" }}>{r.Total_Occurrences.toLocaleString()}</td>
                          <td style={{ padding: "12px 20px", color: "#a1a1aa" }}>{r.Total_Drops.toLocaleString()}</td>
                          <td style={{ padding: "12px 20px", fontWeight: 700, color: risk.color, fontSize: 16 }}>
                            {r.Drop_Percentage.toFixed(1)}%
                          </td>
                          <td style={{ padding: "12px 20px" }}>
                            <span style={{
                              background: risk.bg,
                              color: risk.color,
                              border: `1px solid ${risk.border}`,
                              borderRadius: 6,
                              padding: "3px 10px",
                              fontSize: 11,
                              fontWeight: 700,
                              letterSpacing: 2,
                            }}>
                              {risk.label}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}

        {/* Historical chart */}
        <div style={{
          background: "#18181f",
          border: "1px solid #27272d",
          borderRadius: 12,
          padding: "20px 24px",
          marginTop: 28,
        }}>
          <div style={{ fontSize: 11, color: "#71717a", letterSpacing: 3, textTransform: "uppercase", marginBottom: 18 }}>
            Historical Drop % — All Qualifying CPTs (≥{minOcc} occurrences)
          </div>
          <ResponsiveContainer width="100%" height={240}>
            <BarChart data={filteredRisk} margin={{ top: 0, right: 0, left: -10, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#27272d" vertical={false} />
              <XAxis dataKey="CPT_Code" tick={{ fill: "#71717a", fontSize: 12 }} axisLine={false} tickLine={false} />
              <YAxis tick={{ fill: "#71717a", fontSize: 11 }} axisLine={false} tickLine={false}
                tickFormatter={v => `${v}%`} />
              <Tooltip
                contentStyle={{ background: "#1f1f26", border: "1px solid #3f3f46", borderRadius: 8, fontFamily: "inherit" }}
                labelStyle={{ color: "#a5b4fc", fontWeight: 700 }}
                formatter={(v) => [`${v}%`, "Drop Rate"]}
              />
              <Bar dataKey="Drop_Percentage" radius={[4, 4, 0, 0]}>
                {filteredRisk.map((_, i) => (
                  <Cell key={i} fill={BAR_COLORS[i] ?? "#6366f1"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <p style={{ fontSize: 11, color: "#3f3f46", marginTop: 20, textAlign: "center" }}>
          Plug in your real aggregated risk table by replacing RISK_TABLE at the top of this component.
        </p>
      </div>
    </div>
  );
}
