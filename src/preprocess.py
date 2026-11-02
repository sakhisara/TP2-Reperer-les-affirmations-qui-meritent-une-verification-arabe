"""Code preprocess extrait du notebook. Exécution ordonnée via run.py."""

# %% CELL 4
# Partie : AUDIT
# 2. Définir les indices à compter
PATTERNS = {
    "url": r"https?://[^\s]+|www\.[^\s]+",
    "mention": r"(?<!\w)@[A-Za-z0-9_]+",
    "hashtag": r"(?<!\w)#[\w]+",
    # \d reconnaît aussi les chiffres arabes : ١٢٣…
    "nombre": r"\d+(?:[.,٫٬:/-]\d+)*",
    "retweet": r"(?i)^\s*RT\s+@\w+\s*:"
}

def normaliser_arabe(text):
    """Normalisation pour l'audit, sans modifier les textes originaux."""
    text = unicodedata.normalize("NFKC", str(text))
    text = re.sub(r"[\u064B-\u065F\u0670\u06D6-\u06ED]", "", text)
    text = text.replace("ـ", "")
    text = re.sub("[أإآٱ]", "ا", text)
    text = text.replace("ى", "ي")
    return re.sub(r"\s+", " ", text).strip().lower()

PAYS = [
    "المغرب", "الجزائر", "تونس", "ليبيا", "مصر",
    "السودان", "موريتانيا", "السعودية", "المملكة العربية السعودية",
    "قطر", "الإمارات", "الكويت", "البحرين", "عمان", "سلطنة عمان",
    "اليمن", "العراق", "سوريا", "سورية", "لبنان", "الأردن",
    "فلسطين", "إسرائيل", "إيران", "تركيا",
    "أمريكا", "الولايات المتحدة", "روسيا", "الصين",
    "فرنسا", "بريطانيا", "ألمانيا", "الهند"
]

INSTITUTIONS = [
    "الأمم المتحدة", "مجلس الأمن", "منظمة الصحة العالمية",
    "جامعة الدول العربية", "الاتحاد الأوروبي",
    "البنك الدولي", "صندوق النقد الدولي",
    "وزارة الصحة", "وزارة الداخلية", "وزارة الخارجية",
    "وزارة الدفاع", "البرلمان", "مجلس الوزراء",
    "البيت الأبيض", "المحكمة الجنائية الدولية"
]

def compiler_lexique(terms):
    alternatives = sorted(
        {normaliser_arabe(t) for t in terms},
        key=len, reverse=True
    )
    # Le "_" permet notamment de reconnaître #وزارة_الصحة.
    alternatives = [
        re.escape(t).replace(r"\ ", r"[\s_]+")
        for t in alternatives
    ]
    return re.compile(
        r"(?<!\w)(?:" + "|".join(alternatives) + r")(?!\w)"
    )

pays_pattern = compiler_lexique(PAYS)
institutions_pattern = compiler_lexique(INSTITUTIONS)

def ajouter_indices(df):
    out = df.copy()
    texts = out["tweet_text"]

    for feature, pattern in PATTERNS.items():
        out[f"n_{feature}"] = texts.str.count(pattern)
        out[f"has_{feature}"] = out[f"n_{feature}"].gt(0)

    norm = texts.map(normaliser_arabe)

    out["n_pays"] = norm.map(lambda t: len(pays_pattern.findall(t)))
    out["n_institution"] = norm.map(
        lambda t: len(institutions_pattern.findall(t))
    )

    for feature in ["pays", "institution"]:
        out[f"has_{feature}"] = out[f"n_{feature}"].gt(0)

    return out

train_audit = ajouter_indices(train)
val_audit = ajouter_indices(val)

# %% CELL 9
# Partie : AUDIT
# 4. Repérer les doublons exacts et normalisés
def cle_doublon(text):
    text = normaliser_arabe(text)

    text = re.sub(
        r"^\s*rt\s+@\w+\s*:\s*", "", text, flags=re.I
    )
    text = re.sub(PATTERNS["url"], " URLTOKEN ", text)
    text = re.sub(PATTERNS["mention"], " USERTOKEN ", text)

    return re.sub(r"\s+", " ", text).strip()

def preparer_doublons(df):
    out = df.copy()
    out["cle_exacte"] = out["tweet_text"]
    out["cle_normalisee"] = out["tweet_text"].map(cle_doublon)
    return out

train_audit = preparer_doublons(train_audit)
val_audit = preparer_doublons(val_audit)

def bilan_doublons(df, key):
    groups = (
        df.groupby(key, dropna=False)
        .agg(
            effectif=("class_label", "size"),
            nb_labels=("class_label", "nunique")
        )
    )

    duplicates = groups[groups["effectif"] > 1]

    return {
        "cle": key,
        "groupes_dupliques": len(duplicates),
        "tweets_dans_groupes": int(duplicates["effectif"].sum()),
        "copies_en_exces": int((duplicates["effectif"] - 1).sum()),
        "groupes_labels_contradictoires":
            int((duplicates["nb_labels"] > 1).sum())
    }

display(pd.DataFrame([
    bilan_doublons(train_audit, "cle_exacte"),
    bilan_doublons(train_audit, "cle_normalisee")
]))

# Examiner les groupes normalisés avec labels contradictoires.
nb_labels = train_audit.groupby(
    "cle_normalisee"
)["class_label"].nunique()

cles_contradictoires = nb_labels[nb_labels > 1].index

conflits = train_audit[
    train_audit["cle_normalisee"].isin(cles_contradictoires)
].sort_values("cle_normalisee")

display(conflits[
    ["tweet_id", "tweet_text", "class_label", "cle_normalisee"]
].head(30))
# Sauvegarder le bilan des doublons
doublons_train = pd.DataFrame([
    bilan_doublons(train_audit, "cle_exacte"),
    bilan_doublons(train_audit, "cle_normalisee")
])

doublons_train.to_csv(
    RESULTS / "audit_doublons_train.csv",
    index=False,
    encoding="utf-8-sig"
)

# Sauvegarder tous les conflits, pas seulement les 30 affichés
conflits.to_csv(
    RESULTS / "audit_conflits_normalises.csv",
    index=False,
    encoding="utf-8-sig"
)
print("Bilan des doublons et conflits sauvegardés.")

# %% CELL 17
# Partie : A
# 1-Préparer les données et les imports
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    precision_recall_curve,
    auc
)

RESULTS = Path("/content/results")
RESULTS.mkdir(parents=True, exist_ok=True)

SEED = 42

def presence_nombre_message(text):
    # Retirer uniquement URL et mentions pour cette détection.
    text = re.sub(PATTERNS["url"], " ", text)
    text = re.sub(PATTERNS["mention"], " ", text)
    return bool(re.search(PATTERNS["nombre"], text))

# Copies pour la partie A : les données originales restent disponibles.
train_A = ajouter_indices(train)
val_A = ajouter_indices(val)

for df in [train_A, val_A]:
    df["has_nombre"] = df["tweet_text"].map(presence_nombre_message)
    df["has_pays_ou_institution"] = (
        df["has_pays"] | df["has_institution"]
    )

# Yes = classe positive
y_train = train_A["class_label"].map({"No": 0, "Yes": 1}).to_numpy()
y_val = val_A["class_label"].map({"No": 0, "Yes": 1}).to_numpy()

print("Train :", len(y_train))
print("Validation :", len(y_val))

# %% CELL 55
# Partie : C
# 1. Préparer le vocabulaire et mesurer OOV
import json, random, math
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import precision_recall_curve, precision_score, recall_score, f1_score, auc

OUT_C = RESULTS / 'C'
OUT_C.mkdir(exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CFG = dict(seed=42, epochs=8, batch_size=32)
SEED, EPOCHS, BATCH = CFG['seed'], CFG['epochs'], CFG['batch_size']

def seed_everything(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Même politique que l'audit : contradictions exactes exclues, copies exactes retirées.
labels_C = train.groupby('tweet_text')['class_label'].nunique()
train_fit = train[~train['tweet_text'].isin(labels_C[labels_C > 1].index)]
train_fit = train_fit.drop_duplicates('tweet_text').reset_index(drop=True)
val_C = val.reset_index(drop=True)

def mots_C(t):
    return re.findall(r'https?://\S+|www\.\S+|@\w+|#\w+|\d+(?:[.,٫٬:/-]\d+)*|\w+|[^\w\s]', normaliser_arabe(t))

tw = [mots_C(t) for t in train_fit.tweet_text]
vw = [mots_C(t) for t in val_C.tweet_text]
tc = [list(normaliser_arabe(t)) for t in train_fit.tweet_text]
vc = [list(normaliser_arabe(t)) for t in val_C.tweet_text]

def vocabulaire(sequences):
    freq = Counter(t for seq in sequences for t in seq)
    return {'<PAD>': 0, '<UNK>': 1, **{
        t: i + 2 for i, t in enumerate(sorted(t for t, n in freq.items() if n >= 2))}}

vocab_mots, vocab_chars = vocabulaire(tw), vocabulaire(tc)
LM = int(np.clip(np.ceil(np.percentile(list(map(len, tw)), 95)), 8, 256))
LC = int(np.clip(np.ceil(np.percentile(list(map(len, tc)), 95)), 32, 1024))

def taux_oov(sequences, vocab):
    total = sum(map(len, sequences))
    return sum(t not in vocab for s in sequences for t in s) / max(total, 1)

print('Vocabulaire mots :', len(vocab_mots), '| OOV dev :', taux_oov(vw, vocab_mots))
pd.DataFrame([{'split': 'dev', 'OOV_mots': taux_oov(vw, vocab_mots)}]).to_csv(OUT_C/'oov_dev.csv', index=False)
# En partie F seulement : taux_oov([mots_C(t) for t in test.tweet_text], vocab_mots)

def encoder(sequences, vocab, longueur):
    x = torch.zeros(len(sequences), longueur, dtype=torch.long)
    for i, seq in enumerate(sequences):
        ids = [vocab.get(t, 1) for t in seq[:longueur]] or [1]
        x[i, :len(ids)] = torch.tensor(ids)
    return x

XM, VM = encoder(tw, vocab_mots, LM), encoder(vw, vocab_mots, LM)
XC, VC = encoder(tc, vocab_chars, LC), encoder(vc, vocab_chars, LC)
ytr = torch.tensor(train_fit.class_label.map({'No': 0, 'Yes': 1}).to_numpy(), dtype=torch.float32)
ydev = val_C.class_label.map({'No': 0, 'Yes': 1}).to_numpy()
(OUT_C/'vocabulaire.json').write_text(json.dumps({'mots': vocab_mots, 'caracteres': vocab_chars, 'LM': LM, 'LC': LC}, ensure_ascii=False), encoding='utf-8')

# %% CELL 57
# Partie : C
# 2. Charger AraVec Twitter
from urllib.request import urlretrieve
from zipfile import ZipFile
from gensim.models import Word2Vec

dossier = Path('/content/aravec')
dossier.mkdir(exist_ok=True)
modeles = list(dossier.rglob('full_uni_cbow_300_twitter.mdl'))
if not modeles:
    archive = dossier/'twitter_300.zip'
    if not archive.exists():
        urlretrieve('https://bakrianoo.ewr1.vultrobjects.com/aravec/full_uni_cbow_300_twitter.zip', archive)
    with ZipFile(archive) as z: z.extractall(dossier)
    modeles = list(dossier.rglob('full_uni_cbow_300_twitter.mdl'))
assert len(modeles) == 1, 'Indiquer le chemin du modèle AraVec Twitter 300D.'
av = Word2Vec.load(str(modeles[0])).wv
assert av.vector_size == 300

rng = np.random.default_rng(SEED)
ALEATOIRE = rng.normal(0, .05, (len(vocab_mots), 300)).astype('float32')
ALEATOIRE[0] = 0
ARAVEC = ALEATOIRE.copy()
couverts = set()
for mot in av.index_to_key:
    cle = normaliser_arabe(mot)
    if cle in vocab_mots and vocab_mots[cle] >= 2 and cle not in couverts:
        ARAVEC[vocab_mots[cle]] = av[mot]
        couverts.add(cle)
print('Couverture AraVec :', len(couverts) / max(len(vocab_mots)-2, 1))
del av

# %% CELL 97
# Partie : E
# 5 — Tokeniser les textes
def tokeniser_E(model_id, longueur, anglais=False):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    prep = (
        ArabertPreprocessor(model_name=model_id)
        if "arabert" in model_id
        else None
    )

    def encoder_textes(df, split):
        textes = [
            prep.preprocess(t) if prep else str(t)
            for t in df["tweet_text"]
        ]

        textes = [
            t.strip() or tokenizer.unk_token
            for t in textes
        ]

        ids_complets = tokenizer(
            textes, truncation=False
        )["input_ids"]

        tailles = np.array([
            len(ids) for ids in ids_complets
        ])

        print(
            model_id, split,
            "— textes tronqués :",
            f"{100 * np.mean(tailles > longueur):.2f}%"
        )

        return tokenizer(
            textes,
            padding="max_length",
            truncation=True,
            max_length=longueur,
            return_tensors="pt"
        )

    train_tokens = encoder_textes(AR_TRAIN, "train arabe")
    dev_tokens = encoder_textes(AR_DEV, "dev arabe")

    en_tokens = (
        encoder_textes(EN_TRAIN, "train anglais")
        if anglais else None
    )

    return tokenizer, train_tokens, dev_tokens, en_tokens

# %% CELL 152
# Partie : F3
# F3.1 — Détecter les nombres et les URL
REGEX_URL_F3 = r"https?://[^\s]+|www\.[^\s]+"
REGEX_MENTION_F3 = r"@[A-Za-z0-9_]+"
REGEX_NOMBRE_F3 = r"\d+(?:[.,٫٬:/-]\d+)*"

PROTEGES_F3 = (
    rf"(?:{REGEX_URL_F3})|(?:{REGEX_MENTION_F3})"
)

def contient_nombre_F3(texte):
    contenu = re.sub(PROTEGES_F3, " ", str(texte))
    return bool(re.search(REGEX_NOMBRE_F3, contenu))

def contient_url_F3(texte):
    return bool(re.search(REGEX_URL_F3, str(texte)))

def masquer_nombres_F3(texte):
    # Les URL et mentions sont reconnues avant les nombres.
    motif = rf"{PROTEGES_F3}|(?P<nombre>{REGEX_NOMBRE_F3})"

    return re.sub(
        motif,
        lambda m: " عدد " if m.lastgroup == "nombre" else m.group(),
        str(texte)
    )

exemple = "سجلت الوزارة ١٢٣ حالة https://exemple.com/2024"
print("Original :", exemple)
print("Masqué   :", masquer_nombres_F3(exemple))

# %% CELL 158
# Partie : F3
# F3.4 — Préparer les textes masqués pour le réentraînement
config_F3 = CONFIGS_F1[SEEDS_E[0]]

MODEL_ID_F3 = config_F3["model_id"]
PARTIE_F3 = "E2" if config_F3["deux_tetes"] else "E1"
LAMBDA_F3 = float(config_F3["lambda_en"])
NOM_F3 = f"F3_{MODELE_F1}_nombres_masques"

tokenizer_F3 = AutoTokenizer.from_pretrained(
    OUT_E / f"{MODELE_F1}_seed{SEEDS_E[0]}" / "tokenizer"
)

preprocesseur_F3 = None

if config_F3["preprocessing"] == "ArabertPreprocessor":
    preprocesseur_F3 = ArabertPreprocessor(
        model_name=MODEL_ID_F3
    )

def tokeniser_masque_F3(textes):
    textes = [
        masquer_nombres_F3(t)
        for t in textes
    ]

    if preprocesseur_F3 is not None:
        textes = [
            preprocesseur_F3.preprocess(t)
            for t in textes
        ]

    textes = [
        t.strip() or tokenizer_F3.unk_token
        for t in textes
    ]

    return tokenizer_F3(
        textes,
        padding="max_length",
        truncation=True,
        max_length=config_F3["max_length"],
        return_tensors="pt"
    )

xt_F3 = tokeniser_masque_F3(AR_TRAIN["tweet_text"])
xv_F3 = tokeniser_masque_F3(AR_DEV["tweet_text"])

xe_F3 = None

if PARTIE_F3 == "E2" and LAMBDA_F3 > 0:
    xe_F3 = tokeniser_masque_F3(EN_TRAIN["tweet_text"])

print("Modèle :", MODEL_ID_F3)
print("Expérience :", NOM_F3)

# %% CELL 167
# Partie : F4
# F4.1 — Détecter les entités nommées
from transformers import pipeline

if "ner_F4" not in globals():
    ner_F4 = pipeline(
        "token-classification",
        model="CAMeL-Lab/bert-base-arabic-camelbert-mix-ner",
        aggregation_strategy="simple",
        stride=64,
        device=0 if torch.cuda.is_available() else -1
    )

OUT_F.mkdir(parents=True, exist_ok=True)

# %% CELL 169
# Partie : F4
# F4.2 — Supprimer les indices
def supprimer_nombres_F4(texte):
    motif = rf"{PROTEGES_F3}|(?P<nombre>{REGEX_NOMBRE_F3})"

    return re.sub(
        motif,
        lambda m: " " if m.lastgroup == "nombre" else m.group(),
        str(texte)
    )

def supprimer_entites_F4(texte, entites):
    caracteres = list(str(texte))

    for entite in entites:
        debut = int(entite["start"])
        fin = int(entite["end"])

        caracteres[debut:fin] = [" "] * (fin - debut)

    return "".join(caracteres)

def variantes_F4(texte, entites):
    sans_entites = supprimer_entites_F4(texte, entites)

    return {
        "sans_nombres": supprimer_nombres_F4(texte),
        "sans_entites": sans_entites,
        "sans_nombres_et_entites": supprimer_nombres_F4(sans_entites)
    }

