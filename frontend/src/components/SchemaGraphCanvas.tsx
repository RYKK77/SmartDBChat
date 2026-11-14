import { useEffect, useMemo, useState } from 'react'
import {
  Background,
  Controls,
  Handle,
  MarkerType,
  MiniMap,
  Position,
  ReactFlow,
  type Edge,
  type Node,
  type NodeProps,
} from '@xyflow/react'
import { GitFork, KeyRound, Search, Table2, X } from 'lucide-react'
import { api } from '../services/api'
import type { SchemaColumn, SchemaGraphResponse } from '../types/api'

interface TableNodeData extends Record<string, unknown> {
  label: string
  comment: string
  columns: SchemaColumn[]
  highlighted: boolean
  dimmed: boolean
}

function TableCardNode({ data }: NodeProps<Node<TableNodeData>>) {
  return (
    <div
      className={`er-node ${data.highlighted ? 'highlighted' : ''} ${
        data.dimmed ? 'dimmed' : ''
      }`}
    >
      <Handle type="target" position={Position.Left} style={{ background: 'var(--accent)' }} />
      <div className="er-node-header">
        <span style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
          <Table2 size={13} />
          {data.label}
        </span>
        <span style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 400 }}>
          {data.columns.length} 列
        </span>
      </div>
      {data.comment && (
        <div
          style={{
            padding: '4px 11px',
            fontSize: 11,
            color: 'var(--text-muted)',
            borderBottom: '1px solid var(--border)',
            background: 'var(--bg-surface)',
          }}
        >
          {data.comment}
        </div>
      )}
      <div className="er-node-cols">
        {data.columns.slice(0, 10).map((col) => (
          <div key={col.name} className="er-node-col">
            <span className="er-node-col-name">
              {col.is_primary_key && <KeyRound size={11} color="var(--warning)" />}
              {col.name}
            </span>
            <span className="er-node-col-type">{col.data_type}</span>
          </div>
        ))}
        {data.columns.length > 10 && (
          <div style={{ padding: '2px 11px', fontSize: 11, color: 'var(--text-muted)' }}>
            +{data.columns.length - 10} 更多列...
          </div>
        )}
      </div>
      <Handle type="source" position={Position.Right} style={{ background: 'var(--accent)' }} />
    </div>
  )
}

const nodeTypes = { tableCard: TableCardNode }

interface SchemaGraphCanvasProps {
  graph: SchemaGraphResponse | null
  highlightedTables?: string[]
  selectedTable?: string | null
  onSelectTable?: (tableName: string | null) => void
  onClose?: () => void
  compact?: boolean
}

export function SchemaGraphCanvas({
  graph,
  highlightedTables = [],
  selectedTable,
  onSelectTable,
  onClose,
  compact = false,
}: SchemaGraphCanvasProps) {
  const [sampleRows, setSampleRows] = useState<Record<string, any>[]>([])
  const [loadingSample, setLoadingSample] = useState(false)
  const [inspectCol, setInspectCol] = useState<string>('')
  const [colKeyword, setColKeyword] = useState('')
  const [colValues, setColValues] = useState<any[]>([])
  const [joinTarget, setJoinTarget] = useState<string>('')
  const [joinResult, setJoinResult] = useState<{
    path: string[]
    join_conditions: string[]
  } | null>(null)

  const highlightSet = useMemo(
    () => new Set(highlightedTables.map((t) => t.toLowerCase())),
    [highlightedTables],
  )
  const hasHighlights = highlightSet.size > 0

  const selectedNodeMeta = useMemo(
    () => graph?.nodes.find((n) => n.name === selectedTable) || null,
    [graph, selectedTable],
  )

  useEffect(() => {
    if (!selectedTable || !selectedNodeMeta) return
    setLoadingSample(true)
    setColValues([])
    setJoinResult(null)
    const firstCol = selectedNodeMeta.columns[0]?.name || ''
    setInspectCol(firstCol)
    api
      .sampleTable(selectedTable, 5)
      .then((res) => setSampleRows(res.rows || []))
      .catch(() => setSampleRows([]))
      .finally(() => setLoadingSample(false))
  }, [selectedTable, selectedNodeMeta])

  const handleExploreValues = async () => {
    if (!selectedTable || !inspectCol) return
    try {
      const res = await api.exploreColumnValues(selectedTable, inspectCol, colKeyword, 20)
      setColValues(res.values || [])
    } catch {
      setColValues([])
    }
  }

  const handleFindJoin = async () => {
    if (!selectedTable || !joinTarget) return
    try {
      const res = await api.findJoinPath(selectedTable, joinTarget)
      setJoinResult({ path: res.path, join_conditions: res.join_conditions })
    } catch {
      setJoinResult(null)
    }
  }

  const { nodes, edges } = useMemo(() => {
    if (!graph) return { nodes: [], edges: [] }

    // 网格自动布局（根据表数量自适应列数）
    const colsPerRow = compact ? 2 : 3
    const xGap = 285
    const yGap = 290

    const rfNodes: Node<TableNodeData>[] = graph.nodes.map((n, idx) => {
      const col = idx % colsPerRow
      const row = Math.floor(idx / colsPerRow)
      const isHi = highlightSet.has(n.name.toLowerCase())
      return {
        id: n.name,
        type: 'tableCard',
        position: { x: col * xGap + 24, y: row * yGap + 24 },
        data: {
          label: n.name,
          comment: n.comment,
          columns: n.columns,
          highlighted: isHi,
          dimmed: hasHighlights && !isHi,
        },
      }
    })

    const rfEdges: Edge[] = graph.edges.map((e, idx) => {
      const touchesHighlight =
        hasHighlights &&
        highlightSet.has(e.from_table.toLowerCase()) &&
        highlightSet.has(e.to_table.toLowerCase())
      return {
        id: `e-${idx}-${e.from_table}-${e.to_table}`,
        source: e.from_table,
        target: e.to_table,
        label: `${e.from_column} → ${e.to_column}`,
        animated: touchesHighlight,
        style: {
          stroke: touchesHighlight ? 'var(--accent)' : 'var(--border-strong)',
          strokeWidth: touchesHighlight ? 2.2 : 1.4,
          strokeDasharray: e.inferred ? '5 4' : undefined,
          opacity: hasHighlights && !touchesHighlight ? 0.28 : 0.9,
        },
        labelStyle: { fontSize: 10, fill: 'var(--text-muted)' },
        markerEnd: {
          type: MarkerType.ArrowClosed,
          color: touchesHighlight ? 'var(--accent)' : 'var(--border-strong)',
        },
      }
    })

    return { nodes: rfNodes, edges: rfEdges }
  }, [graph, highlightSet, hasHighlights, compact])

  if (!graph) {
    return (
      <div style={{ padding: 24, color: 'var(--text-muted)', textAlign: 'center' }}>
        连接数据库后即可查看交互式 Schema 关系图谱
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', minHeight: 0 }}>
      <div
        style={{
          padding: '10px 14px',
          borderBottom: '1px solid var(--border)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: 'var(--bg-surface)',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <GitFork size={15} color="var(--accent)" />
          <span style={{ fontWeight: 600, fontSize: 13 }}>Schema 关系图谱</span>
          <span className="pill">
            {graph.table_count} 表 · {graph.edge_count} 关联
          </span>
          {hasHighlights && (
            <span className="pill pill-accent">已高亮 {highlightedTables.length} 张命中表</span>
          )}
        </div>
        {onClose && (
          <button type="button" className="icon-btn" onClick={onClose} title="收起图谱">
            <X size={15} />
          </button>
        )}
      </div>

      <div style={{ flex: 1, minHeight: 260, position: 'relative' }}>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          nodeTypes={nodeTypes}
          fitView
          fitViewOptions={{ padding: 0.16 }}
          onNodeClick={(_, node) => onSelectTable?.(node.id)}
        >
          <Background gap={18} size={1} color="var(--border)" />
          <Controls />
          {!compact && <MiniMap pannable zoomable />}
        </ReactFlow>
      </div>

      {selectedNodeMeta && (
        <div
          style={{
            borderTop: '1px solid var(--border)',
            background: 'var(--bg-surface)',
            padding: 12,
            maxHeight: 290,
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: 10,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Table2 size={14} color="var(--accent)" />
              <strong style={{ fontSize: 13 }}>{selectedNodeMeta.name}</strong>
              {selectedNodeMeta.comment && (
                <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  {selectedNodeMeta.comment}
                </span>
              )}
            </div>
            <button
              type="button"
              className="btn btn-sm btn-ghost"
              onClick={() => onSelectTable?.(null)}
            >
              关闭详情
            </button>
          </div>

          {/* 列值探索 & JOIN 路径查找 */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
            <select
              className="select"
              style={{ width: 'auto', fontSize: 12, padding: '4px 8px' }}
              value={inspectCol}
              onChange={(e) => setInspectCol(e.target.value)}
            >
              {selectedNodeMeta.columns.map((c) => (
                <option key={c.name} value={c.name}>
                  {c.name}
                </option>
              ))}
            </select>
            <input
              className="input"
              style={{ width: 140, fontSize: 12, padding: '4px 8px' }}
              placeholder="关键字筛选值..."
              value={colKeyword}
              onChange={(e) => setColKeyword(e.target.value)}
            />
            <button type="button" className="btn btn-sm" onClick={handleExploreValues}>
              <Search size={12} />
              探测列值
            </button>

            <span style={{ color: 'var(--border-strong)' }}>|</span>

            <select
              className="select"
              style={{ width: 'auto', fontSize: 12, padding: '4px 8px' }}
              value={joinTarget}
              onChange={(e) => setJoinTarget(e.target.value)}
            >
              <option value="">选择目标表查 JOIN 路径...</option>
              {graph.nodes
                .filter((n) => n.name !== selectedNodeMeta.name)
                .map((n) => (
                  <option key={n.name} value={n.name}>
                    {n.name}
                  </option>
                ))}
            </select>
            <button
              type="button"
              className="btn btn-sm"
              disabled={!joinTarget}
              onClick={handleFindJoin}
            >
              <GitFork size={12} />
              最短路径
            </button>
          </div>

          {colValues.length > 0 && (
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, alignItems: 'center' }}>
              <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>去重样例值:</span>
              {colValues.map((v, i) => (
                <span key={i} className="pill pill-info">
                  {String(v)}
                </span>
              ))}
            </div>
          )}

          {joinResult && (
            <div
              style={{
                padding: '6px 10px',
                borderRadius: 'var(--radius-sm)',
                background: 'var(--accent-soft)',
                fontSize: 12,
              }}
            >
              {joinResult.path.length > 0 ? (
                <>
                  <strong>路径:</strong> {joinResult.path.join(' → ')}
                  {joinResult.join_conditions.length > 0 && (
                    <div style={{ fontFamily: 'var(--font-mono)', marginTop: 3 }}>
                      ON {joinResult.join_conditions.join(' AND ')}
                    </div>
                  )}
                </>
              ) : (
                '两表在 Schema 图谱中暂无连通路径'
              )}
            </div>
          )}

          <div>
            <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 4 }}>
              表数据采样 ({loadingSample ? '加载中...' : `${sampleRows.length} 行`})
            </div>
            {sampleRows.length > 0 && (
              <div className="data-table-scroll" style={{ maxHeight: 140 }}>
                <table className="data-table">
                  <thead>
                    <tr>
                      {Object.keys(sampleRows[0]).map((col) => (
                        <th key={col}>{col}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {sampleRows.map((r, i) => (
                      <tr key={i}>
                        {Object.keys(sampleRows[0]).map((col) => (
                          <td key={col}>{String(r[col] ?? '')}</td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  )
}
