import { useState } from 'react'
import { createPersona } from '../api'
import { useToast } from './Toast'
import { OceanEditor } from './OceanRadar'

const ALL_MODULES   = ['gpt', 'speech', 'emotion', 'rag', 'navigation']
const CAPABILITIES  = ['face_recognition', 'emotion_understanding', 'navigation']
const VOICES        = [{ value: 'female', label: 'Female' }, { value: 'male', label: 'Male' }]
const LANGUAGES     = [
  { value: 'en', label: 'English' }, { value: 'es', label: 'Spanish' },
  { value: 'fr', label: 'French'  }, { value: 'de', label: 'German'  },
  { value: 'ja', label: 'Japanese'}, { value: 'zh', label: 'Chinese' },
]
const DEFAULT_TAGS = [
  '[DEFAULT]', '[WAVE]', '[HAPPY]', '[SAD]', '[CONFUSED]',
  '[GREETING]', '[ANGRY]', '[SLEEP]', '[POINT]', '[POSE]', '[SHRUG]',
]

const defaults = {
  name:         '',
  description:  '',
  robot_role:   '',
  allowed_tags: ['[DEFAULT]', '[WAVE]', '[HAPPY]', '[SAD]', '[CONFUSED]'],
  modules:      ['gpt'],
  voice_config: { gender: 'female', language: 'en', rate: '+0%', pitch: '+0Hz' },
  capabilities: { face_recognition: false, emotion_understanding: false, navigation: false },
  personality:  { O: 0.5, C: 0.5, E: 0.5, A: 0.5, N: 0.5 },
}

export default function PersonaModal({ onClose, onCreated }) {
  const [form, setForm]   = useState(defaults)
  const [saving, setSaving] = useState(false)
  const toast = useToast()

  function setField(path, value) {
    setForm(prev => {
      const next = { ...prev }
      const keys = path.split('.')
      let obj = next
      for (let i = 0; i < keys.length - 1; i++) {
        obj[keys[i]] = { ...obj[keys[i]] }
        obj = obj[keys[i]]
      }
      obj[keys[keys.length - 1]] = value
      return next
    })
  }

  function toggleModule(mod) {
    setForm(prev => ({
      ...prev,
      modules: prev.modules.includes(mod)
        ? prev.modules.filter(m => m !== mod)
        : [...prev.modules, mod],
    }))
  }

  function toggleTag(tag) {
    setForm(prev => ({
      ...prev,
      allowed_tags: prev.allowed_tags.includes(tag)
        ? prev.allowed_tags.filter(t => t !== tag)
        : [...prev.allowed_tags, tag],
    }))
  }

  function toggleCapability(cap) {
    setForm(prev => ({
      ...prev,
      capabilities: { ...prev.capabilities, [cap]: !prev.capabilities[cap] },
    }))
  }

  function handleOceanChange(key, value) {
    setForm(prev => ({
      ...prev,
      personality: { ...prev.personality, [key]: value },
    }))
  }

  async function handleSave() {
    if (!form.name.trim())       { toast.error('Name is required');        return }
    if (!form.robot_role.trim()) { toast.error('Role prompt is required'); return }
    setSaving(true)
    try {
      const persona = await createPersona(form)
      toast.success(`Persona '${form.name}' created!`)
      onCreated(persona.persona)
    } catch (e) {
      toast.error(e.message)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="modal-overlay" onClick={e => e.target === e.currentTarget && onClose()}>
      <div className="modal">
        <div className="modal-title">
          New persona
          <button className="modal-close" onClick={onClose}>✕</button>
        </div>

        <div className="form-group">
          <label className="form-label">Name</label>
          <input className="form-input" placeholder="e.g. Security Guard"
            value={form.name} onChange={e => setField('name', e.target.value)} />
        </div>

        <div className="form-group">
          <label className="form-label">Description</label>
          <input className="form-input" placeholder="Short description"
            value={form.description} onChange={e => setField('description', e.target.value)} />
        </div>

        <div className="form-group">
          <label className="form-label">Role prompt</label>
          <textarea className="form-textarea" rows={4}
            placeholder="You are... describe the robot's personality and purpose."
            value={form.robot_role} onChange={e => setField('robot_role', e.target.value)} />
        </div>

        <div className="form-group">
          <label className="form-label">Modules</label>
          <div className="checkbox-group">
            {ALL_MODULES.map(m => (
              <label key={m} className={`checkbox-pill ${form.modules.includes(m) ? 'checked' : ''}`}>
                <input type="checkbox" onChange={() => toggleModule(m)} />{m}
              </label>
            ))}
          </div>
        </div>

        <div className="form-group">
          <label className="form-label">Allowed emotion tags</label>
          <div className="checkbox-group">
            {DEFAULT_TAGS.map(tag => (
              <label key={tag} className={`checkbox-pill ${form.allowed_tags.includes(tag) ? 'checked' : ''}`}>
                <input type="checkbox" onChange={() => toggleTag(tag)} />{tag}
              </label>
            ))}
          </div>
        </div>

        <div className="form-group">
          <label className="form-label">Physical capabilities</label>
          <div className="checkbox-group">
            {CAPABILITIES.map(cap => (
              <label key={cap} className={`checkbox-pill ${form.capabilities[cap] ? 'checked' : ''}`}>
                <input type="checkbox" onChange={() => toggleCapability(cap)} />
                {cap.replace(/_/g, ' ')}
              </label>
            ))}
          </div>
        </div>

        <div className="form-group">
          <label className="form-label">Voice</label>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
            <select className="form-select" value={form.voice_config.gender}
              onChange={e => setField('voice_config.gender', e.target.value)}>
              {VOICES.map(v => <option key={v.value} value={v.value}>{v.label}</option>)}
            </select>
            <select className="form-select" value={form.voice_config.language}
              onChange={e => setField('voice_config.language', e.target.value)}>
              {LANGUAGES.map(l => <option key={l.value} value={l.value}>{l.label}</option>)}
            </select>
          </div>
        </div>

        {/* OCEAN — radar chart + sliders */}
        <div className="form-group">
          <label className="form-label">Personality (OCEAN)</label>
          <OceanEditor values={form.personality} onChange={handleOceanChange} />
        </div>

        <div className="modal-footer">
          <button className="btn btn-secondary" onClick={onClose} disabled={saving}>Cancel</button>
          <button className="btn btn-primary"   onClick={handleSave} disabled={saving}>
            {saving ? 'Saving...' : 'Create persona'}
          </button>
        </div>
      </div>
    </div>
  )
}
