import { useEffect, useState } from 'react'
import { BookOpen, Code2, Plus, Save, Trash2 } from 'lucide-react'
import { api } from '../services/api'
import type { SemanticLayerData } from '../types/api'

interface SemanticLayerStudioProps {
  onNotify: (msg: string, isError?: boolean) => void
}

export function SemanticLayerStudio({ onNotify }: SemanticLayerStudioProps) {
  const [tab, setTab] = useState<'visual' | 'yaml'>('visual')
  const [data, setData] = useState<SemanticLayerData | null>(null)
  const [yamlText, setYamlText] = useState('')
  const [saving, setSaving] = useState(false)

  // 新增表单临时状态
  const [newRule, setNewRule] = useState('')
  const [newMetric, setNewMetric] = useState({ name: '', expression: '', description: '' })
  const [newTerm, setNewTerm] = useState({ term: '', definition: '' })

  const loadSemantic = async () => {
    try {
      const res = await api.getSemanticLayer()
      setData(res.data)
      setYamlText(res.yaml_text)
    } catch (err: any) {
      onNotify(err.message || '加载语义层失败', true)
    }
  }

  useEffect(() => {
    loadSemantic()
  }, [])

  const saveVisual = async (nextData: SemanticLayerData) => {
    setSaving(true)
    try {
      const res = await api.updateSemanticLayer({ data: nextData })
      setData(res.data)
      setYamlText(res.yaml_text)
      onNotify('语义层配置已更新并生效')
    } catch (err: any) {
      onNotify(err.message || '保存语义层失败', true)
    } finally {
      setSaving(false)
    }
  }

  const saveYaml = async () => {
    setSaving(true)
    try {
      const res = await api.updateSemanticLayer({ yaml_text: yamlText })
      setData(res.data)
      setYamlText(res.yaml_text)
      onNotify('YAML 语义层已解析并保存')
    } catch (err: any) {
      onNotify(err.message || 'YAML 格式错误', true)
    } finally {
      setSaving(false)
    }
  }

  if (!data) {
    return <div style={{ padding: 24, color: 'var(--text-muted)' }}>正在加载语义层...</div>
  }

  return (
    <div
      style={{
        flex: 1,
        overflowY: 'auto',
        padding: '20px 28px',
        background: 'var(--bg-app)',
      }}
    >
      <div style={{ maxWidth: 980, margin: '0 auto', display: 'flex', flexDirection: 'column', gap: 16 }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
          <div>
            <h2 style={{ margin: 0, fontSize: 18, fontWeight: 700 }}>业务语义层中心 (Semantic Layer)</h2>
            <p style={{ margin: '4px 0 0', fontSize: 13, color: 'var(--text-secondary)' }}>
              参考 WrenAI MDL 与 SQLBot 术语库：定义指标公式、业务规则（user_rules）、术语映射与 Few-Shot Golden SQL
            </p>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <div className="nav-tabs">
              <button
                type="button"
                className={`nav-tab ${tab === 'visual' ? 'active' : ''}`}
                onClick={() => setTab('visual')}
              >
                <BookOpen size={14} />
                可视化配置
              </button>
              <button
                type="button"
                className={`nav-tab ${tab === 'yaml' ? 'active' : ''}`}
                onClick={() => setTab('yaml')}
              >
                <Code2 size={14} />
                YAML 源码
              </button>
            </div>
            <button
              type="button"
              className="btn btn-primary"
              disabled={saving}
              onClick={() => (tab === 'visual' ? saveVisual(data) : saveYaml())}
            >
              <Save size={14} />
              {saving ? '保存中...' : '保存语义层'}
            </button>
          </div>
        </div>

        {tab === 'yaml' ? (
          <div className="side-section" style={{ padding: 16 }}>
            <textarea
              className="textarea textarea-mono"
              rows={24}
              value={yamlText}
              onChange={(e) => setYamlText(e.target.value)}
            />
          </div>
        ) : (
          <>
            {/* 数据库背景与全局规则 */}
            <div className="side-section" style={{ padding: 16 }}>
              <div className="field-group">
                <label className="field-label">数据库业务背景描述 (Database Description)</label>
                <textarea
                  className="textarea"
                  rows={2}
                  value={data.database_description}
                  onChange={(e) => setData({ ...data, database_description: e.target.value })}
                />
              </div>

              <div className="field-group">
                <label className="field-label">全局业务规则 (User Rules)</label>
                {data.user_rules.map((rule, idx) => (
                  <div key={idx} style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                    <input
                      className="input"
                      value={rule}
                      onChange={(e) => {
                        const next = [...data.user_rules]
                        next[idx] = e.target.value
                        setData({ ...data, user_rules: next })
                      }}
                    />
                    <button
                      type="button"
                      className="icon-btn"
                      onClick={() =>
                        setData({
                          ...data,
                          user_rules: data.user_rules.filter((_, i) => i !== idx),
                        })
                      }
                    >
                      <Trash2 size={14} />
                    </button>
                  </div>
                ))}
                <div style={{ display: 'flex', gap: 8 }}>
                  <input
                    className="input"
                    placeholder="添加新规则，例如：统计营收时排除测试账号..."
                    value={newRule}
                    onChange={(e) => setNewRule(e.target.value)}
                  />
                  <button
                    type="button"
                    className="btn"
                    onClick={() => {
                      if (!newRule.trim()) return
                      setData({ ...data, user_rules: [...data.user_rules, newRule.trim()] })
                      setNewRule('')
                    }}
                  >
                    <Plus size={14} />
                    添加规则
                  </button>
                </div>
              </div>
            </div>

            {/* 指标口径 */}
            <div className="side-section" style={{ padding: 16 }}>
              <div className="side-section-title">
                <span>业务指标口径 ({data.metrics.length})</span>
              </div>
              {data.metrics.map((m, idx) => (
                <div
                  key={idx}
                  style={{
                    border: '1px solid var(--border)',
                    borderRadius: 'var(--radius-sm)',
                    padding: 10,
                    display: 'flex',
                    justifyContent: 'space-between',
                    gap: 10,
                    alignItems: 'center',
                  }}
                >
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontWeight: 600, fontSize: 13 }}>{m.name}</div>
                    <div style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--accent)' }}>
                      {m.expression}
                    </div>
                    {m.description && (
                      <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>{m.description}</div>
                    )}
                  </div>
                  <button
                    type="button"
                    className="icon-btn"
                    onClick={() =>
                      setData({ ...data, metrics: data.metrics.filter((_, i) => i !== idx) })
                    }
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
              <div className="grid-2">
                <input
                  className="input"
                  placeholder="指标名称（如：净消费总额）"
                  value={newMetric.name}
                  onChange={(e) => setNewMetric({ ...newMetric, name: e.target.value })}
                />
                <input
                  className="input textarea-mono"
                  placeholder="SQL 表达式（如：SUM(amount)）"
                  value={newMetric.expression}
                  onChange={(e) => setNewMetric({ ...newMetric, expression: e.target.value })}
                />
              </div>
              <div style={{ display: 'flex', gap: 8 }}>
                <input
                  className="input"
                  placeholder="指标说明（可选）"
                  value={newMetric.description}
                  onChange={(e) => setNewMetric({ ...newMetric, description: e.target.value })}
                />
                <button
                  type="button"
                  className="btn"
                  onClick={() => {
                    if (!newMetric.name.trim() || !newMetric.expression.trim()) return
                    setData({
                      ...data,
                      metrics: [...data.metrics, { ...newMetric }],
                    })
                    setNewMetric({ name: '', expression: '', description: '' })
                  }}
                >
                  <Plus size={14} />
                  新增指标
                </button>
              </div>
            </div>

            {/* 术语与 Golden SQL */}
            <div className="grid-2">
              <div className="side-section" style={{ padding: 16 }}>
                <div className="side-section-title">业务术语映射 ({Object.keys(data.glossary).length})</div>
                {Object.entries(data.glossary).map(([term, defn]) => (
                  <div
                    key={term}
                    style={{
                      display: 'flex',
                      justifyContent: 'space-between',
                      gap: 8,
                      padding: '6px 8px',
                      background: 'var(--bg-subtle)',
                      borderRadius: 'var(--radius-sm)',
                    }}
                  >
                    <div>
                      <strong style={{ fontSize: 12.5 }}>{term}</strong>
                      <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{defn}</div>
                    </div>
                    <button
                      type="button"
                      className="icon-btn"
                      onClick={() => {
                        const copy = { ...data.glossary }
                        delete copy[term]
                        setData({ ...data, glossary: copy })
                      }}
                    >
                      <Trash2 size={13} />
                    </button>
                  </div>
                ))}
                <input
                  className="input"
                  placeholder="业务术语"
                  value={newTerm.term}
                  onChange={(e) => setNewTerm({ ...newTerm, term: e.target.value })}
                />
                <div style={{ display: 'flex', gap: 8 }}>
                  <input
                    className="input"
                    placeholder="等价 SQL 条件或解释"
                    value={newTerm.definition}
                    onChange={(e) => setNewTerm({ ...newTerm, definition: e.target.value })}
                  />
                  <button
                    type="button"
                    className="btn"
                    onClick={() => {
                      if (!newTerm.term.trim() || !newTerm.definition.trim()) return
                      setData({
                        ...data,
                        glossary: { ...data.glossary, [newTerm.term.trim()]: newTerm.definition.trim() },
                      })
                      setNewTerm({ term: '', definition: '' })
                    }}
                  >
                    <Plus size={14} />
                  </button>
                </div>
              </div>

              <div className="side-section" style={{ padding: 16 }}>
                <div className="side-section-title">
                  Few-Shot Golden SQL ({data.golden_examples.length})
                </div>
                {data.golden_examples.map((ex, idx) => (
                  <div
                    key={idx}
                    style={{
                      border: '1px solid var(--border)',
                      borderRadius: 'var(--radius-sm)',
                      padding: 8,
                      display: 'flex',
                      flexDirection: 'column',
                      gap: 4,
                    }}
                  >
                    <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
                      <strong style={{ fontSize: 12.5 }}>{ex.question}</strong>
                      <button
                        type="button"
                        className="icon-btn"
                        onClick={() =>
                          setData({
                            ...data,
                            golden_examples: data.golden_examples.filter((_, i) => i !== idx),
                          })
                        }
                      >
                        <Trash2 size={13} />
                      </button>
                    </div>
                    <pre className="sql-box" style={{ fontSize: 11.5, padding: 8 }}>
                      {ex.sql}
                    </pre>
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  )
}
