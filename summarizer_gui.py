from flask import Flask, render_template_string, request, jsonify
import re
import numpy as np
import networkx as nx
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── NLTK downloads ────────────────────────────────────────────────────────
for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet',
            'averaged_perceptron_tagger', 'averaged_perceptron_tagger_eng']:
    nltk.download(pkg, quiet=True)

# ── NLP Setup ─────────────────────────────────────────────────────────────
STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ── Summarization Functions (same as notebook) ─────────────────────────────
def tfidf_summarizer(text, num_sentences=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text, len(sentences)
    vectorizer   = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(sentences)
    scores       = np.array(tfidf_matrix.sum(axis=1)).flatten()
    ranked       = np.argsort(scores)[::-1]
    selected     = sorted(ranked[:num_sentences])
    summary      = ' '.join([sentences[i] for i in selected])
    return summary, len(selected)

def textrank_summarizer(text, num_sentences=3):
    sentences = sent_tokenize(text)
    if len(sentences) <= num_sentences:
        return text, len(sentences)
    vectorizer = TfidfVectorizer(stop_words='english')
    vectors    = vectorizer.fit_transform(sentences)
    sim_matrix = cosine_similarity(vectors)
    np.fill_diagonal(sim_matrix, 0)
    graph      = nx.from_numpy_array(sim_matrix)
    scores     = nx.pagerank(graph)
    ranked     = sorted(((scores[i], s, i) for i, s in enumerate(sentences)), reverse=True)
    selected   = sorted(ranked[:num_sentences], key=lambda x: x[2])
    summary    = ' '.join([s for _, s, _ in selected])
    return summary, len(selected)

# ── Flask App ──────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Text Summarizer — NLP Project</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: #0f0f1a;
    color: #e8e8f5;
    font-family: 'Segoe UI', sans-serif;
    min-height: 100vh;
  }

  /* ── Header ── */
  .header {
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-bottom: 1px solid #2a2a4a;
    padding: 22px 40px;
    display: flex;
    align-items: center;
    gap: 16px;
  }
  .header h1 {
    font-size: 1.5rem;
    font-weight: 700;
    letter-spacing: -0.5px;
  }
  .header span {
    color: #7c6af7;
  }
  .header p {
    color: #8888aa;
    font-size: 0.82rem;
    margin-top: 2px;
  }

  /* ── Layout ── */
  .container {
    max-width: 1200px;
    margin: 0 auto;
    padding: 32px 24px;
  }

  .grid {
    display: grid;
    grid-template-columns: 1fr 280px;
    gap: 24px;
    align-items: start;
  }

  /* ── Card ── */
  .card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 14px;
    padding: 24px;
  }

  .card-label {
    font-size: 0.7rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #8888aa;
    margin-bottom: 10px;
    font-weight: 600;
  }

  /* ── Textarea ── */
  textarea {
    width: 100%;
    background: #0f0f1a;
    color: #e8e8f5;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 14px 16px;
    font-family: 'Segoe UI', sans-serif;
    font-size: 0.95rem;
    line-height: 1.7;
    resize: vertical;
    min-height: 240px;
    outline: none;
    transition: border-color 0.2s;
  }
  textarea:focus { border-color: #7c6af7; }

  .text-stats {
    text-align: right;
    font-size: 0.8rem;
    color: #8888aa;
    margin-top: 8px;
  }
  .text-stats span { color: #56cfb2; font-weight: 600; }

  /* ── Settings ── */
  .setting-group { margin-bottom: 20px; }

  .setting-label {
    font-size: 0.72rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #8888aa;
    display: block;
    margin-bottom: 10px;
  }

  /* Slider */
  .slider-row {
    display: flex;
    align-items: center;
    gap: 12px;
  }
  input[type=range] {
    flex: 1;
    -webkit-appearance: none;
    height: 4px;
    background: #2a2a4a;
    border-radius: 4px;
    outline: none;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 18px; height: 18px;
    background: #7c6af7;
    border-radius: 50%;
    cursor: pointer;
    transition: background 0.2s;
  }
  input[type=range]::-webkit-slider-thumb:hover { background: #9b8fff; }
  .slider-val {
    font-size: 1.3rem;
    font-weight: 700;
    color: #7c6af7;
    min-width: 24px;
    text-align: center;
  }

  /* Radio */
  .radio-group { display: flex; flex-direction: column; gap: 8px; }
  .radio-option {
    display: flex;
    align-items: center;
    gap: 10px;
    cursor: pointer;
    padding: 8px 12px;
    border-radius: 8px;
    border: 1px solid transparent;
    transition: all 0.2s;
    font-size: 0.9rem;
  }
  .radio-option:hover { border-color: #2a2a4a; background: #0f0f1a; }
  .radio-option input[type=radio] { accent-color: #7c6af7; width: 16px; height: 16px; }
  .radio-option.selected { border-color: #7c6af7; background: rgba(124,106,247,0.08); }

  /* ── Buttons ── */
  .btn-primary {
    width: 100%;
    background: linear-gradient(135deg, #7c6af7, #9b8fff);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 13px;
    font-size: 0.9rem;
    font-weight: 700;
    letter-spacing: 1px;
    cursor: pointer;
    transition: all 0.2s;
    margin-top: 8px;
  }
  .btn-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 24px rgba(124,106,247,0.35);
  }
  .btn-primary:active { transform: translateY(0); }
  .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; transform: none; }

  .btn-secondary {
    width: 100%;
    background: #0f0f1a;
    color: #e8e8f5;
    border: 1px solid #2a2a4a;
    border-radius: 10px;
    padding: 10px;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s;
    margin-top: 8px;
  }
  .btn-secondary:hover { border-color: #7c6af7; color: #7c6af7; }

  /* ── Stats row ── */
  .stats-row {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
  }
  .stat-card {
    background: #1a1a2e;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 16px;
    text-align: center;
    transition: border-color 0.2s;
  }
  .stat-card:hover { border-color: #7c6af7; }
  .stat-val {
    font-size: 1.6rem;
    font-weight: 700;
    color: #56cfb2;
    line-height: 1;
    margin-bottom: 4px;
  }
  .stat-lbl {
    font-size: 0.7rem;
    color: #8888aa;
    text-transform: uppercase;
    letter-spacing: 1px;
  }

  /* ── Tabs ── */
  .tabs { display: flex; gap: 4px; margin-bottom: 16px; }
  .tab-btn {
    padding: 8px 20px;
    border-radius: 8px;
    border: 1px solid #2a2a4a;
    background: #1a1a2e;
    color: #8888aa;
    font-size: 0.82rem;
    cursor: pointer;
    transition: all 0.2s;
    letter-spacing: 0.5px;
  }
  .tab-btn:hover { border-color: #7c6af7; color: #e8e8f5; }
  .tab-btn.active { background: #7c6af7; border-color: #7c6af7; color: white; }

  .tab-panel { display: none; }
  .tab-panel.active { display: block; }

  /* ── Summary Output ── */
  .summary-box {
    background: #0f0f1a;
    border: 1px solid #2a2a4a;
    border-radius: 12px;
    padding: 22px 24px;
    line-height: 1.85;
    font-size: 0.95rem;
    color: #e8e8f5;
    min-height: 100px;
  }
  .summary-header {
    font-size: 0.68rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #7c6af7;
    margin-bottom: 14px;
    padding-bottom: 10px;
    border-bottom: 1px solid #2a2a4a;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }
  .badge {
    background: rgba(124,106,247,0.15);
    color: #7c6af7;
    border: 1px solid rgba(124,106,247,0.3);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.7rem;
  }
  .badge-green {
    background: rgba(86,207,178,0.15);
    color: #56cfb2;
    border-color: rgba(86,207,178,0.3);
  }

  .side-by-side { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

  /* ── Empty State ── */
  .empty-state {
    text-align: center;
    padding: 60px 24px;
    color: #8888aa;
  }
  .empty-state .icon { font-size: 3rem; margin-bottom: 14px; }
  .empty-state p { font-size: 0.85rem; margin-top: 6px; }

  /* ── Loader ── */
  .loader { display: none; text-align: center; padding: 40px; color: #7c6af7; }
  .spinner {
    width: 32px; height: 32px;
    border: 3px solid #2a2a4a;
    border-top-color: #7c6af7;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    margin: 0 auto 12px;
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Copy toast ── */
  .toast {
    position: fixed;
    bottom: 30px; right: 30px;
    background: #56cfb2;
    color: #0f0f1a;
    padding: 12px 22px;
    border-radius: 10px;
    font-weight: 700;
    font-size: 0.85rem;
    opacity: 0;
    transition: opacity 0.3s;
    pointer-events: none;
    z-index: 999;
  }
  .toast.show { opacity: 1; }

  /* ── Divider ── */
  hr { border: none; border-top: 1px solid #2a2a4a; margin: 24px 0; }

  /* ── Responsive ── */
  @media (max-width: 768px) {
    .grid { grid-template-columns: 1fr; }
    .stats-row { grid-template-columns: repeat(2, 1fr); }
    .side-by-side { grid-template-columns: 1fr; }
    .header { padding: 16px 20px; }
    .container { padding: 20px 16px; }
  }
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div>
    <h1>📰 Text Summarizer <span>·</span> NLP Project</h1>
    <p>TF-IDF &amp; TextRank Extractive Summarization &nbsp;·&nbsp; BBC News Dataset</p>
  </div>
</div>

<div class="container">

  <!-- Input + Settings Grid -->
  <div class="grid">

    <!-- Left: Input -->
    <div>
      <div class="card">
        <div class="card-label">Input Text</div>
        <textarea id="inputText"
          placeholder="Paste or type your article here... (minimum 3 sentences recommended)"
          oninput="updateStats()"></textarea>
        <div class="text-stats" id="statsLabel">Words: <span>0</span> &nbsp;|&nbsp; Sentences: <span>0</span></div>
      </div>
    </div>

    <!-- Right: Settings -->
    <div>
      <div class="card">
        <div class="setting-group">
          <span class="setting-label">Method</span>
          <div class="radio-group" id="methodGroup">
            <label class="radio-option selected" onclick="selectMethod(this)">
              <input type="radio" name="method" value="both" checked> Both
            </label>
            <label class="radio-option" onclick="selectMethod(this)">
              <input type="radio" name="method" value="tfidf"> TF-IDF Only
            </label>
            <label class="radio-option" onclick="selectMethod(this)">
              <input type="radio" name="method" value="textrank"> TextRank Only
            </label>
          </div>
        </div>

        <div class="setting-group">
          <span class="setting-label">Summary Sentences</span>
          <div class="slider-row">
            <input type="range" id="nSlider" min="1" max="8" value="3"
                   oninput="document.getElementById('sliderVal').textContent=this.value">
            <div class="slider-val" id="sliderVal">3</div>
          </div>
        </div>

        <button class="btn-primary" onclick="summarize()" id="summarizeBtn">
          ▶ &nbsp; SUMMARIZE
        </button>
        <button class="btn-secondary" onclick="clearAll()">
          🗑 &nbsp; Clear All
        </button>
      </div>

      <!-- Mini info -->
      <div style="margin-top:14px; background:#1a1a2e; border:1px solid #2a2a4a;
                  border-radius:12px; padding:16px; font-size:0.8rem; color:#8888aa;
                  line-height:1.6;">
        <b style="color:#e8e8f5;">TF-IDF</b> scores sentences by term importance weights.<br><br>
        <b style="color:#e8e8f5;">TextRank</b> ranks sentences using a similarity graph and PageRank.
      </div>
    </div>

  </div><!-- /grid -->

  <hr>

  <!-- Output Section -->
  <div id="emptyState" class="empty-state">
    <div class="icon">📝</div>
    <div style="font-size:0.85rem; letter-spacing:1px; text-transform:uppercase;">
      Paste your text and click Summarize
    </div>
    <p>Minimum 3 sentences · Supports any news article or paragraph</p>
  </div>

  <div id="loader" class="loader">
    <div class="spinner"></div>
    <div>Summarizing...</div>
  </div>

  <div id="outputSection" style="display:none;">

    <!-- Stats -->
    <div class="stats-row" id="statsRow">
      <div class="stat-card">
        <div class="stat-val" id="statOrig">—</div>
        <div class="stat-lbl">Original Words</div>
      </div>
      <div class="stat-card">
        <div class="stat-val" id="statSum">—</div>
        <div class="stat-lbl">Summary Words</div>
      </div>
      <div class="stat-card">
        <div class="stat-val" id="statComp">—</div>
        <div class="stat-lbl">Compressed</div>
      </div>
      <div class="stat-card">
        <div class="stat-val" id="statSents">—</div>
        <div class="stat-lbl">Sentences Kept</div>
      </div>
    </div>

    <!-- Tabs -->
    <div class="tabs" id="tabBar">
      <button class="tab-btn active" onclick="switchTab('tfidf', this)">TF-IDF Summary</button>
      <button class="tab-btn" onclick="switchTab('textrank', this)">TextRank Summary</button>
      <button class="tab-btn" onclick="switchTab('both', this)">Side by Side</button>
    </div>

    <!-- Tab: TF-IDF -->
    <div class="tab-panel active" id="tab-tfidf">
      <div class="summary-box">
        <div class="summary-header">
          TF-IDF Summary
          <span class="badge" id="badgeTfidf">— words</span>
        </div>
        <div id="outTfidf"></div>
      </div>
      <button class="btn-secondary" style="margin-top:10px;"
              onclick="copySummary('outTfidf')">📋 &nbsp; Copy</button>
    </div>

    <!-- Tab: TextRank -->
    <div class="tab-panel" id="tab-textrank">
      <div class="summary-box">
        <div class="summary-header">
          TextRank Summary
          <span class="badge badge-green" id="badgeTr">— words</span>
        </div>
        <div id="outTextrank"></div>
      </div>
      <button class="btn-secondary" style="margin-top:10px;"
              onclick="copySummary('outTextrank')">📋 &nbsp; Copy</button>
    </div>

    <!-- Tab: Both -->
    <div class="tab-panel" id="tab-both">
      <div class="side-by-side">
        <div>
          <div class="summary-box">
            <div class="summary-header">
              TF-IDF
              <span class="badge" id="badgeTfidf2">— words</span>
            </div>
            <div id="outTfidf2"></div>
          </div>
          <button class="btn-secondary" style="margin-top:10px;"
                  onclick="copySummary('outTfidf2')">📋 &nbsp; Copy TF-IDF</button>
        </div>
        <div>
          <div class="summary-box">
            <div class="summary-header">
              TextRank
              <span class="badge badge-green" id="badgeTr2">— words</span>
            </div>
            <div id="outTextrank2"></div>
          </div>
          <button class="btn-secondary" style="margin-top:10px;"
                  onclick="copySummary('outTextrank2')">📋 &nbsp; Copy TextRank</button>
        </div>
      </div>
    </div>

  </div><!-- /outputSection -->

</div><!-- /container -->

<!-- Toast -->
<div class="toast" id="toast">Copied!</div>

<script>
  // ── Live stats ──────────────────────────────────────────────────────────
  function updateStats() {
    const text = document.getElementById('inputText').value.trim();
    const words = text ? text.split(/\\s+/).length : 0;
    const sents = text ? (text.match(/[^.!?]+[.!?]+/g) || []).length : 0;
    document.getElementById('statsLabel').innerHTML =
      'Words: <span>' + words + '</span> &nbsp;|&nbsp; Sentences: <span>' + sents + '</span>';
  }

  // ── Method select ───────────────────────────────────────────────────────
  function selectMethod(el) {
    document.querySelectorAll('.radio-option').forEach(r => r.classList.remove('selected'));
    el.classList.add('selected');
  }

  // ── Tab switch ──────────────────────────────────────────────────────────
  function switchTab(name, btn) {
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('tab-' + name).classList.add('active');
    btn.classList.add('active');
  }

  // ── Copy ────────────────────────────────────────────────────────────────
  function copySummary(id) {
    const text = document.getElementById(id).innerText;
    navigator.clipboard.writeText(text).then(() => showToast('Copied!'));
  }

  function showToast(msg) {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 2000);
  }

  // ── Clear ───────────────────────────────────────────────────────────────
  function clearAll() {
    document.getElementById('inputText').value = '';
    document.getElementById('outputSection').style.display = 'none';
    document.getElementById('emptyState').style.display = 'block';
    updateStats();
  }

  // ── Summarize ───────────────────────────────────────────────────────────
  function summarize() {
    const text = document.getElementById('inputText').value.trim();
    if (!text) { alert('Please enter some text first.'); return; }

    const roughSents = (text.match(/[^.!?]+[.!?]+/g) || []).length;
    if (roughSents < 3) {
      alert('Please enter at least 3 sentences for a meaningful summary.');
      return;
    }

    const method = document.querySelector('input[name="method"]:checked').value;
    const n      = parseInt(document.getElementById('nSlider').value);
    const btn    = document.getElementById('summarizeBtn');

    btn.disabled = true;
    document.getElementById('emptyState').style.display  = 'none';
    document.getElementById('outputSection').style.display = 'none';
    document.getElementById('loader').style.display = 'block';

    fetch('/summarize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, method, n_sentences: n })
    })
    .then(r => r.json())
    .then(data => {
      btn.disabled = false;
      document.getElementById('loader').style.display = 'none';

      if (data.error) { alert(data.error); return; }

      // Fill stats
      document.getElementById('statOrig').textContent  = data.orig_words;
      document.getElementById('statSum').textContent   = data.summary_words;
      document.getElementById('statComp').textContent  = data.compression + '%';
      document.getElementById('statSents').textContent = n;

      // Fill outputs
      const tf = data.tfidf_summary  || '(not selected)';
      const tr = data.textrank_summary || '(not selected)';
      const tfW = tf.split(' ').length;
      const trW = tr.split(' ').length;

      document.getElementById('outTfidf').textContent    = tf;
      document.getElementById('outTextrank').textContent = tr;
      document.getElementById('outTfidf2').textContent   = tf;
      document.getElementById('outTextrank2').textContent = tr;

      document.getElementById('badgeTfidf').textContent  = tfW + ' words';
      document.getElementById('badgeTr').textContent     = trW + ' words';
      document.getElementById('badgeTfidf2').textContent = tfW + ' words';
      document.getElementById('badgeTr2').textContent    = trW + ' words';

      // Show correct default tab
      document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      if (method === 'tfidf') {
        document.getElementById('tab-tfidf').classList.add('active');
        document.querySelectorAll('.tab-btn')[0].classList.add('active');
      } else if (method === 'textrank') {
        document.getElementById('tab-textrank').classList.add('active');
        document.querySelectorAll('.tab-btn')[1].classList.add('active');
      } else {
        document.getElementById('tab-both').classList.add('active');
        document.querySelectorAll('.tab-btn')[2].classList.add('active');
      }

      document.getElementById('outputSection').style.display = 'block';
    })
    .catch(err => {
      btn.disabled = false;
      document.getElementById('loader').style.display = 'none';
      alert('Error: ' + err.message);
    });
  }
</script>
</body>
</html>
"""

# ── Routes ─────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/summarize', methods=['POST'])
def summarize():
    data   = request.get_json()
    text   = data.get('text', '').strip()
    method = data.get('method', 'both')
    n      = int(data.get('n_sentences', 3))

    if not text:
        return jsonify({'error': 'No text provided.'})

    sentences = sent_tokenize(text)
    if len(sentences) < 3:
        return jsonify({'error': 'Please provide at least 3 sentences.'})

    result = {'orig_words': len(text.split())}

    tf_summary = tr_summary = None

    if method in ('tfidf', 'both'):
        tf_summary, _ = tfidf_summarizer(text, n)

    if method in ('textrank', 'both'):
        tr_summary, _ = textrank_summarizer(text, n)

    result['tfidf_summary']    = tf_summary
    result['textrank_summary'] = tr_summary

    if   method == 'tfidf':    sw = len(tf_summary.split())
    elif method == 'textrank': sw = len(tr_summary.split())
    else: sw = round((len(tf_summary.split()) + len(tr_summary.split())) / 2)

    result['summary_words'] = sw
    result['compression']   = round((1 - sw / result['orig_words']) * 100, 1)

    return jsonify(result)

# ── Run ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('\n  Text Summarizer running at:  http://127.0.0.1:5000\n')
    app.run(debug=False)
