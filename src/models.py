"""Code models extrait du notebook. Exécution ordonnée via run.py."""

# %% CELL 39
# Partie : B
# 2 — Charger le modèle NLI
NLI_ID = (
    "MoritzLaurer/"
    "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
)

tokenizer_nli = AutoTokenizer.from_pretrained(NLI_ID)

model_nli = AutoModelForSequenceClassification.from_pretrained(
    NLI_ID
).to(DEVICE)

model_nli.eval()

# Identifier la classe entailment sans supposer son indice.
entailment_ids = [
    int(i)
    for i, label in model_nli.config.id2label.items()
    if "entail" in label.lower()
]

assert len(entailment_ids) == 1, model_nli.config.id2label
ENTAILMENT_ID = entailment_ids[0]

print("Labels NLI :", model_nli.config.id2label)

# %% CELL 41
# Partie : B
# 3 — Définir les trois formulations
HYPOTHESES = {
    "ar": {
        "Yes": "تحتوي هذه التغريدة على ادعاء واقعي مهم يستحق التحقق.",
        "No": "تعبر هذه التغريدة عن رأي."
    },
    "fr": {
        "Yes": (
            "Ce tweet contient une affirmation factuelle importante "
            "qu'il faut vérifier."
        ),
        "No": "Ce tweet est une opinion."
    },
    "en": {
        "Yes": (
            "This tweet contains an important factual claim "
            "that deserves fact-checking."
        ),
        "No": "This tweet expresses an opinion."
    }
}

with open(
    RESULTS / "B1_hypotheses.json", "w", encoding="utf-8"
) as f:
    json.dump(HYPOTHESES, f, ensure_ascii=False, indent=2)

# %% CELL 47
# Partie : B
# 6 — Libérer le modèle NLI et charger CAMeLBERT NER
del model_nli, tokenizer_nli
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

NER_ID = "CAMeL-Lab/bert-base-arabic-camelbert-mix-ner"

tokenizer_ner = AutoTokenizer.from_pretrained(
    NER_ID,
    use_fast=True,
    model_max_length=512
)

ner = pipeline(
    "token-classification",
    model=NER_ID,
    tokenizer=tokenizer_ner,
    aggregation_strategy="simple",
    stride=128,
    device=0 if torch.cuda.is_available() else -1
)

print("Modèle NER chargé.")

# %% CELL 59
# Partie : C
# 3. Définir C1, C2, C3 et C4
class AdditiveAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, 1, bias=False)
    def forward(self, h, masque):
        scores = self.v(torch.tanh(self.W(h))).squeeze(-1)
        poids = scores.masked_fill(~masque, -torch.inf).softmax(1)
        return (h * poids.unsqueeze(-1)).sum(1), poids

class ModeleMotsC(nn.Module):
    def __init__(self, arch, matrice, positions=True):
        super().__init__()
        self.arch, self.positions = arch, positions
        self.emb = nn.Embedding.from_pretrained(torch.tensor(matrice.copy()), freeze=False, padding_idx=0)
        self.dropout = nn.Dropout(.3)
        if arch in ['C1', 'C2']:
            self.reseau = nn.LSTM(300, 64, batch_first=True, bidirectional=True)
            if arch == 'C2': self.attention = AdditiveAttention(128)
        else:
            self.projection = nn.Linear(300, 128)
            couche = nn.TransformerEncoderLayer(128, 4, dim_feedforward=256, dropout=.3, batch_first=True)
            self.reseau = nn.TransformerEncoder(couche, 2, enable_nested_tensor=False)
            pe = torch.zeros(LM, 128)
            angle = torch.arange(LM)[:, None] * torch.exp(torch.arange(0,128,2)*(-math.log(10000)/128))
            pe[:, 0::2], pe[:, 1::2] = angle.sin(), angle.cos()
            self.register_buffer('pe', pe)
        self.sortie = nn.Linear(128, 1)

    def forward(self, x, return_attention=False):
        masque = x.ne(0)
        h = self.dropout(self.emb(x))
        poids = None
        if self.arch in ['C1', 'C2']:
            packed = pack_padded_sequence(h, masque.sum(1).cpu(), batch_first=True, enforce_sorted=False)
            sequence, (hidden, _) = self.reseau(packed)
            if self.arch == 'C1': resume = torch.cat([hidden[-2], hidden[-1]], 1)
            else:
                h, _ = pad_packed_sequence(sequence, batch_first=True, total_length=x.size(1))
                resume, poids = self.attention(h, masque)
        else:
            h = self.projection(h)
            if self.positions: h = h + self.pe[:x.size(1)]
            h = self.reseau(h, src_key_padding_mask=~masque)
            resume = (h * masque.unsqueeze(-1)).sum(1) / masque.sum(1, keepdim=True)
        logits = self.sortie(self.dropout(resume)).squeeze(-1)
        return (logits, poids) if return_attention else logits

class CharCNNC(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(len(vocab_chars), 48, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(48, 64, k) for k in [3,4,5]])
        self.sortie = nn.Sequential(nn.Dropout(.3), nn.Linear(192, 1))
    def forward(self, x):
        h = self.emb(x).transpose(1,2)
        features = []
        for conv in self.convs:
            z = torch.relu(conv(h))
            limite = (x.ne(0).sum(1)-conv.kernel_size[0]+1).clamp(min=1)
            masque = torch.arange(z.size(-1), device=x.device)[None,:] < limite[:,None]
            features.append(z.masked_fill(~masque[:,None,:], -torch.inf).amax(-1))
        return self.sortie(torch.cat(features, 1)).squeeze(-1)

# %% CELL 73
# Partie : D
# 1. Préparer AraBERT gelé
from transformers import AutoTokenizer, AutoModel
from arabert.preprocess import ArabertPreprocessor
import joblib

OUT_D = RESULTS / "D"
OUT_D.mkdir(exist_ok=True)

MODEL_D = "aubmindlab/bert-base-arabertv02-twitter"
MAX_LEN_D = 64
SEED_D = CFG["seed"]

seed_everything(SEED_D)

tokenizer_d = AutoTokenizer.from_pretrained(MODEL_D)
preprocessor_d = ArabertPreprocessor(model_name=MODEL_D)

arabert_d = AutoModel.from_pretrained(MODEL_D).to(DEVICE)
arabert_d.requires_grad_(False)
arabert_d.eval()

ytrain_d = ytr.cpu().numpy().astype(int)
ydev_d = ydev

# %% CELL 82
# Partie : D
# 5. D2 — BiLSTM + attention de C2
class ModeleD2(nn.Module):
    def __init__(self, dimension):
        super().__init__()

        self.bilstm = nn.LSTM(
            dimension, 64,
            batch_first=True,
            bidirectional=True
        )

        self.attention = AdditiveAttention(128)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, 1)

    def forward(self, etats, longueurs, return_attention=False):
        sequences = pack_padded_sequence(
            etats, longueurs.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        sorties, _ = self.bilstm(sequences)

        sorties, _ = pad_packed_sequence(
            sorties,
            batch_first=True,
            total_length=etats.size(1)
        )

        masque = (
            torch.arange(etats.size(1), device=etats.device)[None, :]
            < longueurs.to(etats.device)[:, None]
        )

        resume, poids = self.attention(sorties, masque)
        logits = self.classifier(self.dropout(resume)).squeeze(-1)

        return (logits, poids) if return_attention else logits


seed_everything(SEED_D)

train_loader_D = DataLoader(
    TensorDataset(H_train, L_train, ytr),
    batch_size=32,
    shuffle=True
)

dev_loader_D = DataLoader(
    TensorDataset(H_dev, L_dev),
    batch_size=32,
    shuffle=False
)

modele_D2 = ModeleD2(H_train.size(-1)).to(DEVICE)


@torch.inference_mode()
def predire_D2():
    modele_D2.eval()
    scores = []

    for etats, longueurs in dev_loader_D:
        logits = modele_D2(etats.to(DEVICE), longueurs)
        scores.extend(torch.sigmoid(logits).cpu().tolist())

    return np.array(scores)

# %% CELL 99
# Partie : E
# 6 — Définir le modèle commun
class ModeleE(nn.Module):
    def __init__(self, model_id, deux_tetes=False):
        super().__init__()

        self.encodeur = AutoModel.from_pretrained(model_id)
        self.dropout = nn.Dropout(0.1)

        dimension = self.encodeur.config.hidden_size
        langues = ["ar", "en"] if deux_tetes else ["ar"]

        self.tetes = nn.ModuleDict({
            langue: nn.Linear(dimension, 1)
            for langue in langues
        })

    def forward(self, inputs, langue="ar"):
        sorties = self.encodeur(**inputs)

        # Représentation du premier token : [CLS] ou <s>.
        resume = sorties.last_hidden_state[:, 0]

        return self.tetes[langue](
            self.dropout(resume)
        ).squeeze(-1)

