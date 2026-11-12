import React, { useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import {
  BookOpen, FileText, Search, UploadCloud, CheckCircle2, LoaderCircle,
  Sparkles, MessageSquareText, BookMarked, ArrowUpRight, ExternalLink,
  Download, Trash2, ChevronRight, ShieldCheck, Microscope, Quote,
  Menu, X, AlertCircle, RefreshCw
} from 'lucide-react';
import './style.css';

const API = (import.meta.env.VITE_API_URL || 'http://127.0.0.1:8001').replace(/\/$/, '');
const ASK_PATH = import.meta.env.VITE_ASK_ENDPOINT || '/ask';
const PAPERS_PATH = import.meta.env.VITE_SEARCH_ENDPOINT || '/papers';
const HEALTH_PATH = import.meta.env.VITE_HEALTH_ENDPOINT || '/health';
const INGEST_PATH = import.meta.env.VITE_INGEST_ENDPOINT || '/ingest';
const MAX_FILE_SIZE = 20 * 1024 * 1024;

async function readResponse(response) {
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = typeof data.detail === 'string'
      ? data.detail
      : 'Something went wrong. Please try again.';
    throw new Error(detail);
  }
  return data;
}

function App() {
  const [section, setSection] = useState('lab');
  const [mobile, setMobile] = useState(false);
  const [connection, setConnection] = useState('checking');
  const [chunkCount, setChunkCount] = useState(null);
  const [documentCount, setDocumentCount] = useState(null);
  const [files, setFiles] = useState([]);
  const [uploadBusy, setUploadBusy] = useState(false);
  const [uploadMessage, setUploadMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const [paperBusy, setPaperBusy] = useState(false);
  const [error, setError] = useState('');
  const [question, setQuestion] = useState('');
  const [chat, setChat] = useState([]);
  const [researchQuery, setResearchQuery] = useState('');
  const [papers, setPapers] = useState([]);
  const [saved, setSaved] = useState(() => {
    try { return JSON.parse(localStorage.getItem('literaai-saved-papers') || '[]'); }
    catch { return []; }
  });

  const nav = [
    { id: 'lab', label: 'PDF Research Lab', icon: FileText },
    { id: 'chat', label: 'Ask your PDF', icon: MessageSquareText },
    { id: 'discover', label: 'Paper Discovery', icon: Search }
  ];

  async function checkHealth(showError = false) {
    setConnection('checking');
    try {
      const data = await readResponse(await fetch(`${API}${HEALTH_PATH}`));
      setConnection('online');
      setChunkCount(typeof data.chunks === 'number' ? data.chunks : null);
      try {
        const stats = await readResponse(await fetch(`${API}/statistics`));
        setDocumentCount(typeof stats.total_documents === 'number' ? stats.total_documents : null);
        if (typeof stats.total_chunks === 'number') setChunkCount(stats.total_chunks);
      } catch {
        // Health endpoint remains sufficient if statistics is unavailable.
      }
      if (showError) setUploadMessage('Your workspace is up to date.');
      setError('');
    } catch {
      setConnection('offline');
      if (showError) setError('We could not connect right now. Please check that the workspace is running and try again.');
    }
  }

  useEffect(() => { checkHealth(); }, []);
  useEffect(() => {
    localStorage.setItem('literaai-saved-papers', JSON.stringify(saved));
  }, [saved]);

  function chooseFiles(fileList) {
    const incoming = Array.from(fileList || []);
    if (!incoming.length) return;
    const valid = [];
    for (const file of incoming) {
      if (file.type !== 'application/pdf' && !file.name.toLowerCase().endsWith('.pdf')) {
        setError(`${file.name} is not a PDF. Only PDF files can be added.`);
        continue;
      }
      if (file.size > MAX_FILE_SIZE) {
        setError(`${file.name} is larger than the 20 MB limit.`);
        continue;
      }
      if (file.size === 0) {
        setError(`${file.name} is empty.`);
        continue;
      }
      valid.push(file);
    }
    if (valid.length) {
      setFiles(old => {
        const merged = [...old];
        valid.forEach(file => {
          if (!merged.some(existing => existing.name === file.name && existing.size === file.size)) merged.push(file);
        });
        return merged;
      });
      setUploadMessage('');
      setError('');
    }
  }

  async function ingest() {
    if (!files.length || uploadBusy) return;
    setUploadBusy(true);
    setError('');
    setUploadMessage('');
    try {
      const formData = new FormData();
      files.forEach(file => formData.append('files', file));
      const result = await readResponse(await fetch(`${API}${INGEST_PATH}`, {
        method: 'POST',
        body: formData
      }));
      const count = files.length;
      setFiles([]);
      setUploadMessage(result.message || `${count} PDF${count > 1 ? 's' : ''} added successfully. Your documents are ready to explore.`);
      await checkHealth();
    } catch (err) {
      setError(err.message || 'Your PDFs could not be uploaded. Please try again.');
    } finally {
      setUploadBusy(false);
    }
  }

  async function ask(e) {
    e?.preventDefault();
    const q = question.trim();
    if (!q || busy) return;
    setChat(old => [...old, { role: 'user', text: q }]);
    setQuestion('');
    setBusy(true);
    setError('');
    try {
      const data = await readResponse(await fetch(`${API}${ASK_PATH}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, evaluate: true, include_papers: false })
      }));
      setChat(old => [...old, {
        role: 'assistant',
        text: data.answer || 'No answer was returned. Try asking in a different way.',
        sources: Array.isArray(data.sources) ? data.sources : [],
        citationCheck: data.citation_check ?? null,
        evaluation: data.evaluation ?? null
      }]);
    } catch {
      setError('Your question could not be answered right now. Please try again.');
    } finally {
      setBusy(false);
    }
  }

  async function discover(e) {
    e?.preventDefault();
    const q = researchQuery.trim();
    if (!q || paperBusy) return;
    setPaperBusy(true);
    setError('');
    try {
      const data = await readResponse(await fetch(`${API}${PAPERS_PATH}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: q, limit: 10, sort_by: 'relevance' })
      }));
      const list = Array.isArray(data) ? data : (data.papers || data.results || data.works || []);
      setPapers(list);
      if (!list.length) setError('No papers were found for this search. Try a broader topic.');
    } catch {
      setPapers([]);
      setError('Paper discovery is temporarily unavailable. Please try again.');
    } finally {
      setPaperBusy(false);
    }
  }

  const toggleSave = paper => {
    const id = paper.id || paper.doi || paper.title;
    setSaved(old => old.some(x => (x.id || x.doi || x.title) === id)
      ? old.filter(x => (x.id || x.doi || x.title) !== id)
      : [...old, { ...paper, id }]);
  };

  const paperCard = (p, i) => {
    const title = p.title || p.display_name || 'Research paper';
    const doi = p.doi || p.doi_url;
    const pdf = p.pdf_url || p.oa_url;
    const id = p.id || doi || title;
    const authors = Array.isArray(p.authors)
      ? p.authors.map(a => typeof a === 'string' ? a : a.name || a.display_name).join(', ')
      : p.authors || 'Authors unavailable';
    return (
      <article className="paper-card" key={id || i}>
        <div className="paper-meta-top">
          <span>RESEARCH PAPER · {p.publication_year || p.year || 'Year n/a'}</span>
          {(p.is_open_access || p.open_access || p.oa) && <i>OPEN ACCESS</i>}
        </div>
        <h3>{title}</h3>
        <p className="authors">{authors}</p>
        <p>{p.abstract || p.summary || 'Abstract not available.'}</p>
        <div className="paper-foot">
          <span><Quote size={13}/>{p.cited_by_count ?? p.citations ?? '—'} citations</span>
          <button onClick={() => toggleSave({ id, title, authors, year: p.publication_year || p.year, abstract: p.abstract, doi, pdf, url: p.landing_page_url || p.url })}>
            {saved.some(x => x.id === id) ? 'Saved' : 'Save paper'} <BookMarked size={14}/>
          </button>
        </div>
        <div className="paper-links">
          {pdf && <a href={pdf} target="_blank" rel="noreferrer"><Download size={13}/>PDF</a>}
          {doi && <a href={doi.startsWith('http') ? doi : `https://doi.org/${doi}`} target="_blank" rel="noreferrer">DOI <ExternalLink size={12}/></a>}
        </div>
      </article>
    );
  };

  return (
    <div className="layout">
      <aside className={`sidebar ${mobile ? 'open' : ''}`}>
        <a className="logo" href="#">
          <span className="logo-icon"><BookOpen/></span>
          <span>Litera<span>AI</span><small>RESEARCH WORKSPACE</small></span>
        </a>
        <div className="side-label">WORKSPACE</div>
        {nav.map(n => (
          <button key={n.id} className={`nav-item ${section === n.id ? 'active' : ''}`}
            onClick={() => { setSection(n.id); setMobile(false); }}>
            <n.icon size={17}/>{n.label}
            {section === n.id && <ChevronRight size={15} className="nav-chevron"/>}
          </button>
        ))}
        <div className="side-label saved-label">YOUR LIBRARY</div>
        <button className="nav-item" onClick={() => { setSection('saved'); setMobile(false); }}>
          <BookMarked size={17}/>Saved papers <span className="saved-count">{saved.length}</span>
        </button>
        <div className="sidebar-bottom">
          <div className="profile-chip">
            <div className="avatar">L</div>
            <div><strong>LiteraAI Workspace</strong><small>Your research space</small></div>
            <span className="online-dot"/>
          </div>
          <p>Research, refined.</p>
        </div>
      </aside>

      <div className="main-area">
        <header className="topbar">
          <button className="mobile-menu" onClick={() => setMobile(!mobile)}><Menu/></button>
          <div className="breadcrumb">Workspace <span>/</span> {nav.find(n => n.id === section)?.label || 'Saved papers'}</div>
          <div className="top-right">
             
            <button className="icon-btn" title="Refresh workspace" onClick={() => checkHealth(true)}><RefreshCw size={15}/></button>
            <span className="top-avatar">A</span>
          </div>
        </header>

        <main className="content">
          {error && <div className="alert"><AlertCircle size={16}/>{error}<button onClick={() => setError('')}><X size={14}/></button></div>}

          {section === 'lab' && <>
            <div className="page-heading">
              <div><div className="eyebrow">YOUR DOCUMENT INTELLIGENCE</div><h1>PDF Research Lab<span>.</span></h1>
                <p>Bring your research together and explore your documents in one place.</p>
              </div>
              <div className="heading-icon"><Microscope/></div>
            </div>

            <div className="lab-grid">
              <section className="panel upload-panel">
                <div className="panel-title">
                  <div><span className="number">01</span><div><h2>Your document workspace</h2><p>Your indexed research is ready to explore.</p></div></div>
                  <span className="optional">LIBRARY</span>
                </div>
                <div className="workflow-list">
                  <div><span><CheckCircle2/></span><section><strong>Your research workspace</strong>
                    <small>Your documents are ready to explore and search.</small></section></div>
                  <div><span><FileText/></span><section><strong>Your research library</strong>
                    <small>{documentCount !== null ? `${documentCount} document${documentCount === 1 ? '' : 's'} · ` : ''}{chunkCount !== null ? `${chunkCount} searchable sections` : 'Your documents will appear here after upload.'}</small></section></div>
                </div>
                <button className="action-button" onClick={() => checkHealth(true)}><RefreshCw size={16}/> Refresh workspace <ArrowUpRight size={16}/></button>
              </section>

              <section className="panel workflow-panel">
                <div className="panel-title"><div><span className="number">02</span><div><h2>Add PDFs</h2><p>Upload one or more documents to your library.</p></div></div></div>
                <div className={`upload-box ${files.length ? 'selected' : ''}`}>
                  <input type="file" accept=".pdf,application/pdf" multiple
                    onChange={e => { chooseFiles(e.target.files); e.target.value = ''; }}/>
                  {files.length ? <>
                    <div className="pdf-icon"><FileText/></div>
                    <div className="upload-text"><strong>{files.length} PDF{files.length > 1 ? 's' : ''} selected</strong>
                      <small>{files.map(f => f.name).join(' · ')}</small></div>
                    <button className="icon-btn" title="Clear selected files" onClick={() => setFiles([])}><Trash2 size={16}/></button>
                  </> : <>
                    <div className="upload-symbol"><UploadCloud/></div><strong>Choose PDF files</strong>
                    <p>Click to browse · PDF only · up to 20 MB each</p>
                  </>}
                </div>
                <button className="outline-button" disabled={!files.length || uploadBusy} onClick={ingest}>
                  {uploadBusy ? <><LoaderCircle className="spin"/> Adding documents…</> : <><UploadCloud size={15}/> Add to my library</>}
                </button>
                {uploadMessage && <div className="upload-success"><CheckCircle2 size={15}/>{uploadMessage}</div>}
                <div className="privacy"><ShieldCheck size={14}/> Your existing documents stay in your library.</div>
              </section>
            </div>

            <section className="insights">
              <div className="insight-header"><div><span className="complete"><CheckCircle2 size={15}/> READY WHEN YOU ARE</span><h2>Explore your documents</h2></div>
                <button className="subtle-btn" onClick={() => setSection('chat')}>Ask questions <ArrowUpRight size={14}/></button></div>
              <p>Ask questions, find important details, and review evidence from your uploaded research.</p>
            </section>
          </>}

          {section === 'chat' && <>
            <div className="page-heading"><div><div className="eyebrow">DOCUMENT Q&A</div><h1>Ask your PDF<span>.</span></h1>
              <p>Get clear answers grounded in the documents in your library.</p></div><div className="heading-icon"><MessageSquareText/></div></div>
            <div className="chat-panel">
                <div className="chat-context"><span className="context-file"><FileText size={15}/>Your research library</span><span><i/> Ready</span></div>
                <div className="chat-messages">
                  {chat.length === 0 && <div className="chat-welcome"><div className="chat-welcome-icon"><Sparkles/></div>
                    <h2>Let's explore your research.</h2><p>Ask about findings, methodology, assumptions, limitations or evidence.</p>
                    <div className="suggested">
                      <button onClick={() => setQuestion('Summarise the main contribution of the indexed research.')}>Summarise the main contribution</button>
                      <button onClick={() => setQuestion('What methodology is described in the indexed documents?')}>Explain the methodology</button>
                      <button onClick={() => setQuestion('What limitations or research gaps are discussed?')}>Identify limitations and gaps</button>
                    </div></div>}
                  {chat.map((m, i) => <div className={`message ${m.role}`} key={i}>
                    <div className="message-avatar">{m.role === 'assistant' ? <Sparkles size={15}/> : 'A'}</div>
                    <div className="message-body"><strong>{m.role === 'assistant' ? 'LiteraAI' : 'You'}</strong><p>{m.text}</p>
                      {m.sources?.length > 0 && <div className="sources"><b>Sources</b>{m.sources.map((s, j) => <span key={j}><Quote size={12}/><span><strong>{s.source || 'Document'} · Page {s.page ?? '—'}</strong><br/>{s.text || ''}</span></span>)}</div>}
                      {m.citationCheck && <div className="sources"><b>Source check</b><span>{m.citationCheck.valid ? '✓ ' : '⚠ '}{m.citationCheck.message || (m.citationCheck.valid ? 'Sources verified' : 'Please review the cited sources.')}</span></div>}
                      {m.evaluation && <div className="sources"><b>Answer quality</b><span>Supported: {m.evaluation.supported ?? '—'} · Unsupported: {m.evaluation.unsupported ?? '—'} · Contradicted: {m.evaluation.contradicted ?? '—'}</span></div>}
                    </div></div>)}
                  {busy && <div className="message assistant"><div className="message-avatar"><Sparkles size={15}/></div><div className="message-body"><strong>LiteraAI</strong><p><LoaderCircle className="spin"/> Reviewing your documents…</p></div></div>}
                </div>
                <form className="chat-input" onSubmit={ask}><input value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask a question about your PDFs…"/>
                  <button disabled={!question.trim() || busy}>{busy ? <LoaderCircle className="spin"/> : <ArrowUpRight/>}</button></form>
                <p className="chat-disclaimer">Answers include document references so you can check the original source.</p>
              </div>
          </>}

          {section === 'discover' && <>
            <div className="page-heading"><div><div className="eyebrow">ACADEMIC LITERATURE SEARCH</div><h1>Paper Discovery<span>.</span></h1>
              <p>Discover relevant research and build your reading list.</p></div><div className="heading-icon"><Search/></div></div>
            <section className="panel discovery-panel"><form onSubmit={discover}><label>What would you like to explore?</label>
              <div className="discover-input"><Search/><input value={researchQuery} onChange={e => setResearchQuery(e.target.value)} placeholder="e.g. Recent advances in retrieval-augmented generation"/>
                <button disabled={!researchQuery.trim() || paperBusy}>{paperBusy ? <LoaderCircle className="spin"/> : <ArrowUpRight/>}</button></div></form></section>
            {papers.length > 0 && <><div className="results-title"><h2>Research results <span>({papers.length})</span></h2></div><div className="paper-grid">{papers.map(paperCard)}</div></>}
          </>}

          {section === 'saved' && <>
            <div className="page-heading"><div><div className="eyebrow">YOUR READING LIST</div><h1>Saved papers<span>.</span></h1>
              <p>Your saved research papers, stored in this browser.</p></div><div className="heading-icon"><BookMarked/></div></div>
            {saved.length ? <div className="paper-grid">{saved.map((p, i) => <article className="paper-card" key={p.id || i}>
              <h3>{p.title}</h3><p className="authors">{p.authors || ''} · {p.year || ''}</p><p>{p.abstract || ''}</p>
              <button className="remove-saved" onClick={() => toggleSave(p)}>Remove from saved <Trash2 size={14}/></button>
            </article>)}</div> : <div className="empty-workspace"><BookMarked/><h2>Your reading list is empty</h2>
              <p>Save papers from Paper Discovery to collect them here.</p><button className="action-button compact" onClick={() => setSection('discover')}>Discover papers <ArrowUpRight size={15}/></button></div>}
          </>}

          <footer className="footer"><span>LiteraAI · Research, refined.</span><span>Academic discovery workspace</span></footer>
        </main>
      </div>
    </div>
  );
}

createRoot(document.getElementById('root')).render(<App/>);
