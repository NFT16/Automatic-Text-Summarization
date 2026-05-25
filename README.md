# Automatic Text Summarization

An NLP-based extractive text summarization system with a Flask web interface. Supports two algorithms — **TF-IDF** and **TextRank** — with a side-by-side comparison view.

---

## How It Works

This is an **extractive summarizer**: it selects the most important sentences from the original text rather than generating new ones.

**TF-IDF** (Term Frequency–Inverse Document Frequency): scores each sentence by the combined weight of its words. Words that are frequent in the document but rare across general text get higher scores. The top-scoring sentences form the summary.

**TextRank**: builds a graph where each sentence is a node. Edges between nodes are weighted by cosine similarity (how alike two sentences are). PageRank is then run on the graph to find the most "central" sentences.

---

## Project Structure

```
├── Automatic Text Summarization.ipynb   # Exploration and development notebook
├── summarizer_gui.py                    # Flask app with embedded HTML/CSS/JS UI
├── bbc-text.csv                         # BBC News dataset used for testing
└── requirements.txt                     # Python dependencies
```

---

## Features

- Two summarization algorithms: TF-IDF and TextRank
- Adjustable summary length (1–8 sentences)
- Side-by-side comparison of both methods
- Live word/sentence count on input
- Compression ratio stats
- Copy-to-clipboard for each summary
- Responsive dark-themed UI

---

## Setup

**Requirements:** Python 3.8+

```bash
# 1. Clone the repo
git clone https://github.com/NFT16/Automatic-Text-Summarization.git
cd Automatic-Text-Summarization

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
python summarizer_gui.py
```

Then open your browser at: `http://127.0.0.1:5000`

NLTK data (punkt, stopwords, wordnet, etc.) downloads automatically on first run.

---

## Usage

1. Paste any text (minimum 3 sentences) into the input box.
2. Choose a method: **TF-IDF**, **TextRank**, or **Both**.
3. Set the number of sentences for the summary using the slider.
4. Click **Summarize**.
5. View results in individual tabs or side-by-side. Copy any summary with one click.

---

## Tech Stack

| Library | Purpose |
|---|---|
| Flask | Web server and API |
| NLTK | Tokenization, stopword removal, lemmatization |
| scikit-learn | TF-IDF vectorization, cosine similarity |
| NetworkX | Graph construction for TextRank / PageRank |
| NumPy | Matrix operations |

---

## Dataset

`bbc-text.csv` contains BBC News articles across 5 categories (business, entertainment, politics, sport, tech). Used for testing and benchmarking the summarizer in the notebook.

---

## Limitations

- **Extractive only**: summaries are stitched sentences from the original text. No paraphrasing or abstraction.
- **English only**: preprocessing (stopwords, lemmatization) is configured for English.
- **Short texts**: fewer than 3 sentences will be returned as-is without summarization.

---

## License

This project is for educational purposes. Dataset credit: [BBC articles fulltext and category](https://www.kaggle.com/datasets/yufengdev/bbc-fulltext-and-category).
