"""
Run:
    python web_app.py
Open in browser: http://127.0.0.1:5000


"""

import os
import re
import unicodedata
from collections import Counter
from flask import Flask, request, render_template_string

# PyTorch and model imports
import torch
import torch.nn as nn

# ----------------- Minimal tokenizer fallbacks -----------------
class BasicTokenizer:
    """A tiny tokenizer for Bengali fallback. Splits on whitespace and punctuation."""
    def tokenize(self, text: str):
        # keep Bengali letters, basic punctuation, split others
        tokens = re.findall(r"[\u0980-\u09FF]+|[A-Za-z]+|[0-9]+|\S", text)
        return tokens

try:
    import spacy
    _spacy_nlp = None
except Exception:
    _spacy_nlp = None

# ----------------- Preprocessing classes (adapted) -----------------
class ProcessBengaliCorpus:
    def __init__(self) -> None:
        self.data = None
        self.tokenizer = BasicTokenizer()

    def clean_data(self, data):
        self.data = list(map(lambda x: re.sub(r"[a-zA-Z0-9\()\_\-]", "", x), data))
        self.data = list(map(lambda x: re.sub(r"\s*\u09cd\s*", "\u09cd", x), self.data))
        self.data = list(map(lambda x: re.sub(r"\s+\u09cd", "", x), self.data))
        self.data = list(map(lambda x: re.sub(r"\u09cd\s+", "", x), self.data))
        self.data = list(map(lambda x: re.sub(r"\s+", " ", x), self.data))
        self.data = [i.replace("ঃ", ":") for i in self.data]
        self.data = [i.replace("।", ".") for i in self.data]
        self.data = [i.strip() for i in self.data]
        self.data = [unicodedata.normalize("NFC", i) for i in self.data]
        return self.data

    def tokenize_bengla(self, sen):
        tokenized = self.tokenizer.tokenize(sen.strip())
        token = [str(t).strip() for t in tokenized if str(t).strip()]
        return token

class ProcessEnglishCorpus:
    def __init__(self) -> None:
        self.data = None
        if _spacy_nlp is not None:
            try:
                self.eng_tokenizer = spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])
            except Exception:
                self.eng_tokenizer = None
        else:
            self.eng_tokenizer = None

    def clean_data(self, data):
        self.data = [i.lower() for i in data]
        self.data = list(map(lambda x: re.sub(r"[^\w\s\.\?\!,']", "", x), self.data))
        self.data = list(map(lambda x: re.sub(r"\s+", " ", x), self.data))
        self.data = [i.strip() for i in self.data]
        return self.data

    def tokenize_english(self, text):
        if self.eng_tokenizer is None:
            # simple fallback
            return re.findall(r"\w+|[\.!?,'\"]", text.lower())
        tokenized = self.eng_tokenizer(text.strip())
        token = [t.text.lower() for t in tokenized]
        return token

# Instantiate preprocessors
bn_proc = ProcessBengaliCorpus()
en_proc = ProcessEnglishCorpus()

# ----------------- Vocabulary -----------------
class Vocab:
    def __init__(self):
        self.specials = ["<pad>", "<sos>", "<eos>", "<unk>"]

    def vocab_builder(self, data, max_size=30000, min_freq=2):
        counter = Counter()
        for sent in data:
            counter.update(sent)
        words = [w for w, f in counter.items() if f > min_freq and w not in self.specials]
        words = sorted(words, key=lambda w: counter[w], reverse=True)
        if max_size:
            words = words[: max_size - len(self.specials)]
        self.itos = list(self.specials) + words
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        self.pad_idx = self.stoi["<pad>"]
        self.sos_idx = self.stoi["<sos>"]
        self.eos_idx = self.stoi["<eos>"]
        self.unk_idx = self.stoi["<unk>"]

    def get_itos(self):
        return self.itos

    def get_stoi(self):
        return self.stoi

    def encode(self, tokens, add_eos=True):
        ids = [self.stoi.get(t, self.unk_idx) for t in tokens]
        if add_eos:
            ids.append(self.eos_idx)
        return ids

    def decode(self, ids):
        out = []
        for i in ids:
            tok = self.itos[i] if 0 <= i < len(self.itos) else "<unk>"
            if tok in ("<eos>", "<pad>"):
                break
            if tok == "<sos>":
                continue
            out.append(tok)
        return out

# ----------------- Transformer model components (adapted) -----------------
class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, dropout, device):
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        if mask is not None:
            energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(self.dropout(attention), V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x, attention

class PositionwiseFeedforwardLayer(nn.Module):
    def __init__(self, hid_dim, pf_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(hid_dim, pf_dim)
        self.fc2 = nn.Linear(pf_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x

class EncoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask):
        _src, _ = self.self_attention(src, src, src, src_mask)
        src = self.self_attn_layer_norm(src + self.dropout(_src))
        _src = self.feedforward(src)
        src = self.ff_layer_norm(src + self.dropout(_src))
        return src

class Encoder(nn.Module):
    def __init__(self, input_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=100):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(input_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([EncoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(self.device)

    def forward(self, src, src_mask):
        batch_size = src.shape[0]
        src_len = src.shape[1]
        pos = torch.arange(0, src_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        src = self.dropout((self.tok_embedding(src) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            src = layer(src, src_mask)
        return src

class DecoderLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.self_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.enc_attn_layer_norm = nn.LayerNorm(hid_dim)
        self.ff_layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttentionLayer(hid_dim, n_heads, dropout, device)
        self.feedforward = PositionwiseFeedforwardLayer(hid_dim, pf_dim, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        _trg, _ = self.self_attention(trg, trg, trg, trg_mask)
        trg = self.self_attn_layer_norm(trg + self.dropout(_trg))
        _trg, attention = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        trg = self.enc_attn_layer_norm(trg + self.dropout(_trg))
        _trg = self.feedforward(trg)
        trg = self.ff_layer_norm(trg + self.dropout(_trg))
        return trg, attention

class Decoder(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device, max_length=100):
        super().__init__()
        self.device = device
        self.tok_embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_embedding = nn.Embedding(max_length, hid_dim)
        self.layers = nn.ModuleList([DecoderLayer(hid_dim, n_heads, pf_dim, dropout, device) for _ in range(n_layers)])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([hid_dim])).to(device)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        pos = torch.arange(0, trg_len).unsqueeze(0).repeat(batch_size, 1).to(self.device)
        trg = self.dropout((self.tok_embedding(trg) * self.scale) + self.pos_embedding(pos))
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        output = self.fc_out(trg)
        return output, attention

class Seq2SeqTransformer(nn.Module):
    def __init__(self, encoder, decoder, src_pad_idx, trg_pad_idx, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask

    def make_trg_mask(self, trg):
        trg_pad_mask = (trg != self.trg_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_len = trg.shape[1]
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=self.device)).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        output, attention = self.decoder(trg, enc_src, trg_mask, src_mask)
        return output, attention

# ----------------- Utility & loading (attempt to reuse user's checkpoint logic) -----------------

def initialize_weights(m):
    if hasattr(m, "weight") and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

# Paths and device
CKPT_PATH = os.path.join(os.getcwd(), "saved_models", "transformer_checkpoint.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
src_vocab = Vocab()
tgt_vocab = Vocab()
loaded_ok = False

if os.path.exists(CKPT_PATH):
    try:
        ckpt = torch.load(CKPT_PATH, map_location=device)
        h = ckpt["hparams"]
        enc = Encoder(h["input_dim"], h["hid_dim"], h["enc_layers"], h["enc_heads"], h["enc_pf_dim"], h["enc_dropout"], device)
        dec = Decoder(h["output_dim"], h["hid_dim"], h["dec_layers"], h["dec_heads"], h["dec_pf_dim"], h["dec_dropout"], device)
        model = Seq2SeqTransformer(enc, dec, h["src_pad_idx"], h["trg_pad_idx"], device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)
        model.eval()
        src_vocab.itos = ckpt.get("src_itos", [])
        tgt_vocab.itos = ckpt.get("tgt_itos", [])
        src_vocab.stoi = {w: i for i, w in enumerate(src_vocab.itos)}
        tgt_vocab.stoi = {w: i for i, w in enumerate(tgt_vocab.itos)}
        if src_vocab.itos and tgt_vocab.itos:
            src_vocab.pad_idx = src_vocab.stoi.get("<pad>")
            src_vocab.sos_idx = src_vocab.stoi.get("<sos>")
            src_vocab.eos_idx = src_vocab.stoi.get("<eos>")
            src_vocab.unk_idx = src_vocab.stoi.get("<unk>")
            tgt_vocab.pad_idx = tgt_vocab.stoi.get("<pad>")
            tgt_vocab.sos_idx = tgt_vocab.stoi.get("<sos>")
            tgt_vocab.eos_idx = tgt_vocab.stoi.get("<eos>")
            tgt_vocab.unk_idx = tgt_vocab.stoi.get("<unk>")
            loaded_ok = True
    except Exception as e:
        print("Failed to load checkpoint:", e)
else:
    print(f"Checkpoint not found at {CKPT_PATH}. App will use fallback translation.")

# ----------------- Translate function -----------------

def translate_sentence(sentence, model, src_vocab, tgt_vocab, device, max_len=50):
    # If model isn't loaded, return a naive fallback (word-by-word copy)
    if model is None or not loaded_ok:
        # fallback: return input words reversed as a dummy BN-like string
        toks = en_proc.tokenize_english(sentence)
        return " ".join(toks[::-1]), None

    model.eval()
    tokens = en_proc.tokenize_english(sentence)
    tokens = [src_vocab.sos_idx] + src_vocab.encode(tokens, add_eos=False) + [src_vocab.eos_idx]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)

    trg_indexes = [tgt_vocab.sos_idx]
    attention = None

    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_token = output.argmax(2)[:, -1].item()
        trg_indexes.append(pred_token)
        if pred_token == tgt_vocab.eos_idx:
            break

    trg_tokens = tgt_vocab.decode(trg_indexes)
    return " ".join(trg_tokens), attention

# ----------------- Flask app -----------------
app = Flask(__name__)

HTML = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>EN → BN Translator</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
  </head>
  <body class="bg-light">
    <div class="container py-5">
      <div class="card shadow-sm">
        <div class="card-body">
          <h3 class="card-title mb-3">English → Bengali Translator</h3>
          <form method="post" action="/translate">
            <div class="mb-3">
              <label for="entext" class="form-label">Enter English text</label>
              <textarea class="form-control" id="entext" name="entext" rows="3">{{ entext or '' }}</textarea>
            </div>
            <button type="submit" class="btn btn-primary">Translate</button>
          </form>
          {% if translated is not none %}
          <hr>
          <h5>Result</h5>
          <p><strong>EN:</strong> {{ entext }}</p>
          <p><strong>BN:</strong> {{ translated }}</p>
          {% if not loaded_ok %}
          <div class="alert alert-warning" role="alert">
            Model not loaded. Showing fallback output. Place your trained checkpoint at <code>./app/saved_models/transformer_checkpoint.pt</code> for real translations.
          </div>
          {% endif %}
          {% endif %}
        </div>
      </div>
    </div>
  </body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML, entext=None, translated=None, loaded_ok=loaded_ok)

@app.route('/translate', methods=['POST'])
def do_translate():
    entext = request.form.get('entext', '')
    translated, attention = translate_sentence(entext, model, src_vocab, tgt_vocab, device)
    return render_template_string(HTML, entext=entext, translated=translated, loaded_ok=loaded_ok)

if __name__ == '__main__':
    # Ensure the directory for checkpoint exists message
    print(f"Checkpoint path: {CKPT_PATH}")
    app.run(debug=True)
