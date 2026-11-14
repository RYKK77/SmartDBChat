import { useMemo, useState } from 'react'
import { ArrowDown, ArrowUp, Download, Search } from 'lucide-react'

interface DataTableViewProps {
  rows: Record<string, any>[]
  fileName?: string
}

export function DataTableView({ rows, fileName = 'smartdbchat_result.csv' }: DataTableViewProps) {
  const [search, setSearch] = useState('')
  const [sortCol, setSortCol] = useState<string | null>(null)
  const [sortAsc, setSortAsc] = useState(true)
  const [page, setPage] = useState(0)
  const pageSize = 25

  const columns = useMemo(() => (rows.length > 0 ? Object.keys(rows[0]) : []), [rows])

  const filteredRows = useMemo(() => {
    const q = search.trim().toLowerCase()
    let list = q
      ? rows.filter((r) =>
          columns.some((c) => String(r[c] ?? '').toLowerCase().includes(q)),
        )
      : [...rows]

    if (sortCol) {
      list.sort((a, b) => {
        const va = a[sortCol]
        const vb = b[sortCol]
        if (va === vb) return 0
        if (va === null || va === undefined) return 1
        if (vb === null || vb === undefined) return -1
        if (typeof va === 'number' && typeof vb === 'number') {
          return sortAsc ? va - vb : vb - va
        }
        return sortAsc
          ? String(va).localeCompare(String(vb), 'zh-CN')
          : String(vb).localeCompare(String(va), 'zh-CN')
      })
    }
    return list
  }, [rows, columns, search, sortCol, sortAsc])

  const totalPages = Math.max(1, Math.ceil(filteredRows.length / pageSize))
  const safePage = Math.min(page, totalPages - 1)
  const pageRows = filteredRows.slice(safePage * pageSize, (safePage + 1) * pageSize)

  const handleSort = (col: string) => {
    if (sortCol === col) {
      setSortAsc(!sortAsc)
    } else {
      setSortCol(col)
      setSortAsc(true)
    }
  }

  const exportCsv = () => {
    const escapeCell = (val: any) => {
      const str = val === null || val === undefined ? '' : String(val)
      return `"${str.replace(/"/g, '""')}"`
    }
    const header = columns.map(escapeCell).join(',')
    const lines = filteredRows.map((r) => columns.map((c) => escapeCell(r[c])).join(','))
    const csv = '\uFEFF' + [header, ...lines].join('\n')
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = fileName
    a.click()
    URL.revokeObjectURL(url)
  }

  if (!rows || rows.length === 0) {
    return <div style={{ color: 'var(--text-muted)', fontSize: 13 }}>暂无数据</div>
  }

  return (
    <div className="data-table-wrap">
      <div className="data-table-toolbar">
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, maxWidth: 300 }}>
          <Search size={14} color="var(--text-muted)" />
          <input
            className="input"
            style={{ padding: '4px 8px', fontSize: 12.5 }}
            placeholder="过滤当前结果..."
            value={search}
            onChange={(e) => {
              setSearch(e.target.value)
              setPage(0)
            }}
          />
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
            共 {filteredRows.length} 行
          </span>
          <button type="button" className="btn btn-sm" onClick={exportCsv}>
            <Download size={13} />
            导出 CSV
          </button>
        </div>
      </div>

      <div className="data-table-scroll">
        <table className="data-table">
          <thead>
            <tr>
              {columns.map((col) => (
                <th key={col} onClick={() => handleSort(col)}>
                  <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                    {col}
                    {sortCol === col && (sortAsc ? <ArrowUp size={12} /> : <ArrowDown size={12} />)}
                  </span>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row, idx) => (
              <tr key={idx}>
                {columns.map((col) => (
                  <td key={col} title={String(row[col] ?? '')}>
                    {row[col] === null || row[col] === undefined ? (
                      <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>NULL</span>
                    ) : (
                      String(row[col])
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {totalPages > 1 && (
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '8px 12px',
            borderTop: '1px solid var(--border)',
            fontSize: 12,
            color: 'var(--text-muted)',
          }}
        >
          <span>
            第 {safePage + 1} / {totalPages} 页
          </span>
          <div style={{ display: 'flex', gap: 6 }}>
            <button
              type="button"
              className="btn btn-sm"
              disabled={safePage === 0}
              onClick={() => setPage((p) => Math.max(0, p - 1))}
            >
              上一页
            </button>
            <button
              type="button"
              className="btn btn-sm"
              disabled={safePage >= totalPages - 1}
              onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
            >
              下一页
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
