import { useMemo, useState } from 'react'
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  PolarAngleAxis,
  PolarGrid,
  PolarRadiusAxis,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import { BarChart3 } from 'lucide-react'
import type { ChartConfig, ChartItem } from '../types/api'

const PALETTE = [
  '#0f766e',
  '#0284c7',
  '#d97706',
  '#059669',
  '#2563eb',
  '#db2777',
  '#65a30d',
  '#ea580c',
]

interface ChartStudioProps {
  rows: Record<string, any>[]
  llmChartConfig?: ChartConfig | null
  recommendedChartConfig?: ChartConfig | null
}

function isNumericValue(v: any): boolean {
  return typeof v === 'number' && !Number.isNaN(v)
}

export function KpiStrip({ rows }: { rows: Record<string, any>[] }) {
  const kpis = useMemo(() => {
    if (!rows || rows.length === 0) return []
    const cols = Object.keys(rows[0])
    if (rows.length === 1) {
      return cols.slice(0, 6).map((col) => ({
        label: col,
        value: rows[0][col],
      }))
    }
    // 对多行结果自动计算数值列汇总 KPI（最多展示 3 个）
    const numCols = cols.filter((c) => {
      const lower = c.toLowerCase()
      if (lower === 'id' || lower.endsWith('_id')) return false
      return rows.some((r) => isNumericValue(r[c]))
    })
    return numCols.slice(0, 3).map((col) => {
      const vals = rows.map((r) => r[col]).filter(isNumericValue) as number[]
      const sum = vals.reduce((a, b) => a + b, 0)
      const avg = vals.length ? sum / vals.length : 0
      return {
        label: `${col} (均值 / 总计)`,
        value: `${avg.toFixed(2)} / ${Number.isInteger(sum) ? sum : sum.toFixed(2)}`,
      }
    })
  }, [rows])

  if (kpis.length === 0) return null

  return (
    <div className="kpi-grid">
      {kpis.map((item) => (
        <div key={item.label} className="kpi-card">
          <div className="kpi-label">{item.label}</div>
          <div className="kpi-value">{String(item.value ?? '-')}</div>
        </div>
      ))}
    </div>
  )
}

function buildChartDataFromItem(
  item: ChartItem,
  fallbackSeriesName?: string,
): {
  data: Record<string, any>[]
  seriesKeys: string[]
  xKey: string
  showLegend: boolean
} {
  const d = item.data || {}
  if (item.type === 'line' && Array.isArray(d.series) && d.series.length > 0) {
    const firstX: any[] = d.series[0]?.x || d.labels || []
    const rows = firstX.map((xVal, idx) => {
      const row: Record<string, any> = { label: String(xVal) }
      d.series.forEach((s: any, sIdx: number) => {
        const key = s.name || `系列${sIdx + 1}`
        row[key] = Number(s.y?.[idx] ?? 0)
      })
      return row
    })
    return {
      data: rows,
      seriesKeys: d.series.map((s: any, idx: number) => s.name || `系列${idx + 1}`),
      xKey: 'label',
      showLegend: true,
    }
  }

  if (item.type === 'radar' && Array.isArray(d.series) && d.series.length > 0) {
    const labels: any[] = d.labels || []
    const rows = labels.map((lbl, idx) => {
      const row: Record<string, any> = { label: String(lbl) }
      d.series.forEach((s: any, sIdx: number) => {
        const key = s.name || `维度${sIdx + 1}`
        row[key] = Number(s.values?.[idx] ?? 0)
      })
      return row
    })
    return {
      data: rows,
      seriesKeys: d.series.map((s: any, idx: number) => s.name || `维度${idx + 1}`),
      xKey: 'label',
      showLegend: true,
    }
  }

  if (item.type === 'scatter' && Array.isArray(d.x) && Array.isArray(d.y)) {
    const rows = d.x.map((xv: any, idx: number) => ({
      x: Number(xv ?? 0),
      y: Number(d.y[idx] ?? 0),
      name: d.hover_names?.[idx] ?? `点 ${idx + 1}`,
    }))
    return { data: rows, seriesKeys: ['y'], xKey: 'x', showLegend: false }
  }

  const metricName =
    (typeof d.y_axis_title === 'string' && d.y_axis_title.trim()) ||
    (typeof d.series_name === 'string' && d.series_name.trim()) ||
    (typeof d.metric_name === 'string' && d.metric_name.trim()) ||
    (fallbackSeriesName && fallbackSeriesName.trim()) ||
    ''

  const seriesKey = metricName || 'value'
  const labels: any[] = d.labels || []
  const values: any[] = d.values || []
  const rows = labels.map((lbl, idx) => ({
    label: String(lbl),
    [seriesKey]: Number(values[idx] ?? 0),
  }))
  return {
    data: rows,
    seriesKeys: [seriesKey],
    xKey: 'label',
    showLegend: Boolean(metricName),
  }
}

export function ChartStudio({
  rows,
  llmChartConfig,
  recommendedChartConfig,
}: ChartStudioProps) {
  const columns = useMemo(() => (rows.length > 0 ? Object.keys(rows[0]) : []), [rows])
  const numericColumns = useMemo(
    () =>
      columns.filter((c) => {
        const lower = c.toLowerCase()
        if (lower === 'id' || lower.endsWith('_id')) return false
        return rows.some((r) => isNumericValue(r[c]))
      }),
    [columns, rows],
  )
  const labelColumns = useMemo(
    () => columns.filter((c) => !numericColumns.includes(c)),
    [columns, numericColumns],
  )

  const initialPreset = llmChartConfig?.charts?.[0] || recommendedChartConfig?.charts?.[0] || null
  const [sourceMode, setSourceMode] = useState<'auto' | 'custom'>(
    initialPreset ? 'auto' : 'custom',
  )
  const [chartType, setChartType] = useState<string>(initialPreset?.type || 'bar')
  const [xAxis, setXAxis] = useState<string>(labelColumns[0] || columns[0] || '')
  const [yAxis, setYAxis] = useState<string>(numericColumns[0] || columns[1] || columns[0] || '')

  if (!rows || rows.length < 2 || columns.length < 2 || numericColumns.length === 0) {
    return null
  }

  const presetCharts =
    (llmChartConfig?.charts?.length ? llmChartConfig.charts : recommendedChartConfig?.charts) || []

  const activeItem: ChartItem =
    sourceMode === 'auto' && presetCharts.length > 0
      ? presetCharts[0]
      : {
          type: chartType,
          title: `${yAxis} 按 ${xAxis} 分布`,
          description: '自定义交互式图表配置',
          data:
            chartType === 'scatter'
              ? {
                  x: rows.slice(0, 80).map((r) => Number(r[xAxis] ?? 0)),
                  y: rows.slice(0, 80).map((r) => Number(r[yAxis] ?? 0)),
                  hover_names: rows.slice(0, 80).map((r) => String(r[labelColumns[0] || xAxis] ?? '')),
                }
              : {
                  labels: rows.slice(0, 50).map((r) => String(r[xAxis] ?? '')),
                  values: rows.slice(0, 50).map((r) => Number(r[yAxis] ?? 0)),
                },
        }

  const effectiveType = sourceMode === 'auto' ? activeItem.type : chartType

  const renderChartBody = (item: ChartItem, typeOverride: string) => {
    const { data, seriesKeys, xKey, showLegend } = buildChartDataFromItem(
      { ...item, type: typeOverride },
      sourceMode === 'custom' ? yAxis : undefined,
    )
    return (
      <div style={{ width: '100%', height: 310 }}>
        <ResponsiveContainer width="100%" height="100%">
          {typeOverride === 'pie' ? (
            <PieChart>
              <Pie
                data={data}
                dataKey={seriesKeys[0] || 'value'}
                nameKey={xKey}
                cx="50%"
                cy="50%"
                innerRadius={56}
                outerRadius={105}
                paddingAngle={2}
                label={({ name, percent }) => `${name} (${((percent ?? 0) * 100).toFixed(1)}%)`}
              >
                {data.map((_, index) => (
                  <Cell key={`cell-${index}`} fill={PALETTE[index % PALETTE.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          ) : typeOverride === 'line' ? (
            <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey={xKey} tick={{ fontSize: 12, fill: 'var(--text-muted)' }} />
              <YAxis tick={{ fontSize: 12, fill: 'var(--text-muted)' }} />
              <Tooltip />
              {showLegend && <Legend />}
              {seriesKeys.map((k, idx) => (
                <Line
                  key={k}
                  type="monotone"
                  dataKey={k}
                  stroke={PALETTE[idx % PALETTE.length]}
                  strokeWidth={2.5}
                  dot={{ r: 3.5 }}
                />
              ))}
            </LineChart>
          ) : typeOverride === 'area' ? (
            <AreaChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey={xKey} tick={{ fontSize: 12, fill: 'var(--text-muted)' }} />
              <YAxis tick={{ fontSize: 12, fill: 'var(--text-muted)' }} />
              <Tooltip />
              {showLegend && <Legend />}
              {seriesKeys.map((k, idx) => (
                <Area
                  key={k}
                  type="monotone"
                  dataKey={k}
                  stroke={PALETTE[idx % PALETTE.length]}
                  fill={PALETTE[idx % PALETTE.length]}
                  fillOpacity={0.22}
                />
              ))}
            </AreaChart>
          ) : typeOverride === 'scatter' ? (
            <ScatterChart margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis
                type="number"
                dataKey="x"
                name={(sourceMode === 'auto' && item.data?.x_axis_title) || xAxis}
                tick={{ fontSize: 12, fill: 'var(--text-muted)' }}
              />
              <YAxis
                type="number"
                dataKey="y"
                name={(sourceMode === 'auto' && item.data?.y_axis_title) || yAxis}
                tick={{ fontSize: 12, fill: 'var(--text-muted)' }}
              />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Scatter name={item.title || `${xAxis} vs ${yAxis}`} data={data} fill={PALETTE[0]} />
            </ScatterChart>
          ) : typeOverride === 'radar' ? (
            <RadarChart cx="50%" cy="50%" outerRadius="76%" data={data}>
              <PolarGrid stroke="var(--border)" />
              <PolarAngleAxis dataKey={xKey} tick={{ fontSize: 11, fill: 'var(--text-secondary)' }} />
              <PolarRadiusAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
              {seriesKeys.map((k, idx) => (
                <Radar
                  key={k}
                  name={k}
                  dataKey={k}
                  stroke={PALETTE[idx % PALETTE.length]}
                  fill={PALETTE[idx % PALETTE.length]}
                  fillOpacity={0.25}
                />
              ))}
              <Legend />
              <Tooltip />
            </RadarChart>
          ) : (
            <BarChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
              <XAxis dataKey={xKey} tick={{ fontSize: 12, fill: 'var(--text-muted)' }} />
              <YAxis tick={{ fontSize: 12, fill: 'var(--text-muted)' }} />
              <Tooltip />
              {showLegend && <Legend />}
              {seriesKeys.map((k, idx) => (
                <Bar
                  key={k}
                  dataKey={k}
                  fill={PALETTE[idx % PALETTE.length]}
                  radius={[4, 4, 0, 0]}
                />
              ))}
            </BarChart>
          )}
        </ResponsiveContainer>
      </div>
    )
  }

  return (
    <div className="chart-studio">
      <div className="chart-toolbar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <BarChart3 size={16} color="var(--accent)" />
          <div>
            <div style={{ fontWeight: 600, fontSize: 13 }}>
              {activeItem.title || '数据可视化'}
            </div>
            {activeItem.description && (
              <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
                {activeItem.description}
              </div>
            )}
          </div>
        </div>

        <div className="chart-controls">
          {presetCharts.length > 0 && (
            <button
              type="button"
              className={`btn btn-sm ${sourceMode === 'auto' ? 'btn-primary' : ''}`}
              onClick={() => setSourceMode('auto')}
            >
              智能推荐
            </button>
          )}
          <select
            className="select"
            aria-label="图表类型"
            value={sourceMode === 'auto' ? activeItem.type : chartType}
            onChange={(e) => {
              setChartType(e.target.value)
              setSourceMode('custom')
            }}
          >
            <option value="bar">柱状图 (Bar)</option>
            <option value="line">折线图 (Line)</option>
            <option value="area">面积图 (Area)</option>
            <option value="pie">饼图 (Pie)</option>
            <option value="scatter">散点图 (Scatter)</option>
            <option value="radar">雷达图 (Radar)</option>
          </select>

          <select
            className="select"
            aria-label="X轴维度"
            value={xAxis}
            onChange={(e) => {
              setXAxis(e.target.value)
              setSourceMode('custom')
            }}
          >
            {columns.map((c) => (
              <option key={c} value={c}>
                X: {c}
              </option>
            ))}
          </select>

          <select
            className="select"
            aria-label="Y轴指标"
            value={yAxis}
            onChange={(e) => {
              setYAxis(e.target.value)
              setSourceMode('custom')
            }}
          >
            {numericColumns.map((c) => (
              <option key={c} value={c}>
                Y: {c}
              </option>
            ))}
          </select>
        </div>
      </div>

      {sourceMode === 'auto' && presetCharts.length > 1 ? (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          {presetCharts.map((item, idx) => (
            <div
              key={`${item.type}-${item.title || idx}`}
              style={
                idx > 0
                  ? { borderTop: '1px solid var(--border)', paddingTop: 14 }
                  : undefined
              }
            >
              {idx > 0 && (
                <div style={{ marginBottom: 8 }}>
                  <div style={{ fontWeight: 600, fontSize: 13 }}>
                    {item.title || `图表 ${idx + 1}`}
                  </div>
                  {item.description && (
                    <div style={{ fontSize: 11.5, color: 'var(--text-muted)' }}>
                      {item.description}
                    </div>
                  )}
                </div>
              )}
              {renderChartBody(item, item.type)}
            </div>
          ))}
        </div>
      ) : (
        renderChartBody(activeItem, effectiveType)
      )}
    </div>
  )
}
