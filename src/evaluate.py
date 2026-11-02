"""Code evaluate extrait du notebook. Exécution ordonnée via run.py."""

# %% CELL 7
# Partie : AUDIT
# 3. Produire le tableau des comptes
FEATURES = [
    "url", "mention", "hashtag", "nombre",
    "pays", "institution", "retweet"
]

def tableau_indices(df, split):
    rows = []

    for feature in FEATURES:
        rows.append({
            "split": split,
            "indice": feature,
            "occurrences": int(df[f"n_{feature}"].sum()),
            "tweets_concernes": int(df[f"has_{feature}"].sum()),
            "pourcentage": round(
                100 * df[f"has_{feature}"].mean(), 2
            )
        })

    return pd.DataFrame(rows)

audit_indices = pd.concat([
    tableau_indices(train_audit, "train"),
    tableau_indices(val_audit, "dev")
], ignore_index=True)

display(audit_indices)
audit_indices.to_csv(
    RESULTS / "audit_indices.csv",
    index=False,
    encoding="utf-8-sig"
)
print("Comptage des indices sauvegardé.")

# %% CELL 11
# Partie : AUDIT
# 5. Vérifier le chevauchement train–dev
def chevauchement(reference, evaluation):
    rows = []

    for key in ["cle_exacte", "cle_normalisee"]:
        mask = evaluation[key].isin(set(reference[key]))

        rows.append({
            "cle": key,
            "tweets_evaluation_presents_dans_train": int(mask.sum()),
            "pourcentage_evaluation": round(100 * mask.mean(), 2)
        })

    return pd.DataFrame(rows)

display(chevauchement(train_audit, val_audit))

# En partie F seulement :
# display(chevauchement(train_audit, test_audit))
chevauchement(train_audit, val_audit).to_csv(
    RESULTS / "audit_chevauchement_train_dev.csv",
    index=False,
    encoding="utf-8-sig"
)
print("Chevauchement train–dev sauvegardé.")

# %% CELL 13
# Partie : AUDIT
# 6. Détecter les quasi-doublons dans le train
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# Une ligne par texte normalisé : les doublons sont déjà traités plus haut.
unique_train = (
    train_audit
    .drop_duplicates("cle_normalisee")
    .reset_index(drop=True)
)

vectorizer_audit = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    min_df=2,
    dtype=np.float32
)

# Ajustement sur train uniquement.
X = vectorizer_audit.fit_transform(unique_train["cle_normalisee"])

neighbors = NearestNeighbors(
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
).fit(X)

SEUIL_SIMILARITE = 0.90
pairs = []

# Traitement par lots pour limiter la mémoire.
for start in range(0, len(unique_train), 256):
    end = min(start + 256, len(unique_train))

    distances, indices = neighbors.radius_neighbors(
        X[start:end],
        radius=1 - SEUIL_SIMILARITE,
        return_distance=True
    )

    for local_i, (dists, ids) in enumerate(zip(distances, indices)):
        i = start + local_i

        for distance, j in zip(dists, ids):
            if j <= i:
                continue

            a = unique_train.iloc[i]
            b = unique_train.iloc[j]

            pairs.append({
                "id_1": a["tweet_id"],
                "id_2": b["tweet_id"],
                "texte_1": a["tweet_text"],
                "texte_2": b["tweet_text"],
                "label_1": a["class_label"],
                "label_2": b["class_label"],
                "similarite": float(1 - distance)
            })

quasi_doublons = pd.DataFrame(pairs, columns=[
    "id_1", "id_2", "texte_1", "texte_2",
    "label_1", "label_2", "similarite"
]).sort_values("similarite", ascending=False)

print("Paires candidates :", len(quasi_doublons))
display(quasi_doublons.head(20))
# Sauvegarder toutes les paires, pas seulement les 20 affichées
quasi_doublons.to_csv(
    RESULTS / "audit_quasi_doublons.csv",
    index=False,
    encoding="utf-8-sig"
)
print("Quasi-doublons sauvegardés.")

# %% CELL 14
# Partie : AUDIT
# 6. Détecter les quasi-doublons dans le train
journal = RESULTS / "results.csv"

if not journal.exists():
    pd.DataFrame(columns=[
        "partie", "modele", "split", "seed", "seuil",
        "f1_yes", "pr_auc", "macro_f1",
        "temps_entrainement_s", "notes"
    ]).to_csv(
        journal,
        index=False,
        encoding="utf-8-sig"
    )

print("Journal prêt :", journal)

# %% CELL 19
# Partie : A
# 2 — Distribution des labels
def distribution_labels(df, split):
    subsets = {
        "tous": df,
        "avec_nombre": df[df["has_nombre"]],
        "sans_nombre": df[~df["has_nombre"]]
    }

    rows = []

    for groupe, subset in subsets.items():
        n = len(subset)
        yes = int(subset["class_label"].eq("Yes").sum())

        rows.append({
            "split": split,
            "groupe": groupe,
            "effectif": n,
            "Yes": yes,
            "No": n - yes,
            "pourcentage_Yes": 100 * yes / n if n else np.nan
        })

    return pd.DataFrame(rows)

distribution_A = pd.concat([
    distribution_labels(train_A, "train"),
    distribution_labels(val_A, "dev"),
    distribution_labels(
        pd.concat([train_A, val_A], ignore_index=True),
        "train+dev"
    )
], ignore_index=True)

display(distribution_A.round(2))

distribution_A.to_csv(
    RESULTS / "A_distribution_labels.csv",
    index=False,
    encoding="utf-8-sig"
)

# %% CELL 21
# Partie : A
# 3 — Mesurer les indices associés à Yes
rows = []

for split, df in [("train", train_A), ("dev", val_A)]:
    for indice in [
        "has_nombre", "has_url", "has_pays_ou_institution"
    ]:
        for presence in [False, True]:
            subset = df[df[indice].eq(presence)]

            rows.append({
                "split": split,
                "indice": indice,
                "present": presence,
                "effectif": len(subset),
                "pourcentage_Yes": (
                    100 * subset["class_label"].eq("Yes").mean()
                    if len(subset) else np.nan
                )
            })

indices_labels_A = pd.DataFrame(rows)

display(indices_labels_A.round(2))

indices_labels_A.to_csv(
    RESULTS / "A_indices_labels.csv",
    index=False,
    encoding="utf-8-sig"
)

# %% CELL 23
# Partie : A
# 4 — Calculer les métriques et enregistrer les scores
def evaluer_A(modele, predictions, scores, seuil=np.nan,
              temps_entrainement=0.0, notes=""):

    predictions = np.asarray(predictions, dtype=int)
    scores = np.asarray(scores, dtype=float)

    precision_curve, recall_curve, _ = precision_recall_curve(
        y_val, scores
    )

    ligne = {
        "partie": "A",
        "modele": modele,
        "split": "dev",
        "seed": SEED,
        "seuil": seuil,
        "precision_yes": precision_score(
            y_val, predictions, zero_division=0
        ),
        "rappel_yes": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1_yes": f1_score(
            y_val, predictions, zero_division=0
        ),
        "pr_auc": auc(recall_curve, precision_curve),
        "macro_f1": f1_score(
            y_val, predictions,
            average="macro", zero_division=0
        ),
        "accuracy": accuracy_score(y_val, predictions),
        "temps_entrainement_s": temps_entrainement,
        "notes": notes
    }

    chemin = RESULTS / "results.csv"

    if chemin.exists():
        historique = pd.read_csv(chemin)
    else:
        historique = pd.DataFrame()

    # Éviter de dupliquer une expérience lors d'une réexécution.
    if not historique.empty:
        meme_experience = (
            historique["partie"].eq("A")
            & historique["modele"].eq(modele)
            & historique["split"].eq("dev")
            & historique["seed"].eq(SEED)
        )
        historique = historique.loc[~meme_experience]

    nouvelle_ligne = pd.DataFrame([ligne])

    if historique.empty:
        historique = nouvelle_ligne
    else:
        historique = pd.concat(
            [historique, nouvelle_ligne],
            ignore_index=True
        )

    historique.to_csv(
        chemin, index=False, encoding="utf-8-sig"
    )

    display(nouvelle_ligne.round(4))
    return ligne

# %% CELL 25
# Partie : A
# 5 — Évaluer « toujours No » et les trois règles
# 1. Toujours No
pred_no = np.zeros(len(y_val), dtype=int)

evaluer_A(
    "Toujours_No",
    predictions=pred_no,
    scores=pred_no,
    notes="Toutes les prédictions sont No."
)

# 2. Règles demandées
regles = {
    "Regle_nombre": "has_nombre",
    "Regle_URL": "has_url",
    "Regle_pays_institution": "has_pays_ou_institution"
}

for nom, colonne in regles.items():
    predictions = val_A[colonne].astype(int).to_numpy()

    evaluer_A(
        nom,
        predictions=predictions,
        scores=predictions,
        seuil=0.5,
        notes=f"Yes si {colonne}=True ; score binaire."
    )

# %% CELL 29
# Partie : A
# 7 — Choisir le seuil sur dev
precision_curve, recall_curve, seuils = precision_recall_curve(
    y_val, scores_val_A
)

# Le dernier point de la courbe n'a pas de seuil associé.
p = precision_curve[:-1]
r = recall_curve[:-1]

f1_par_seuil = np.divide(
    2 * p * r,
    p + r,
    out=np.zeros_like(p),
    where=(p + r) != 0
)

candidats = np.flatnonzero(
    np.isclose(f1_par_seuil, f1_par_seuil.max())
)

meilleur_indice = candidats[
    np.argmin(np.abs(seuils[candidats] - 0.5))
]

seuil_A = float(seuils[meilleur_indice])
pred_val_A = (scores_val_A >= seuil_A).astype(int)

print(f"Seuil choisi sur dev : {seuil_A:.6f}")

evaluer_A(
    "TFIDF_LR_seuil_optimise",
    predictions=pred_val_A,
    scores=scores_val_A,
    seuil=seuil_A,
    temps_entrainement=temps_A,
    notes="Même modèle ; seuil maximisant F1-Yes sur dev."
)

courbe_seuils_A = pd.DataFrame({
    "seuil": seuils,
    "f1_yes": f1_par_seuil
})

courbe_seuils_A.to_csv(
    RESULTS / "A_seuils_dev.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(7, 4))
plt.plot(seuils, f1_par_seuil)
plt.axvline(
    seuil_A, color="red", linestyle="--",
    label=f"Seuil choisi : {seuil_A:.3f}"
)
plt.xlabel("Seuil")
plt.ylabel("F1 de Yes sur dev")
plt.title("Choix du seuil sur la validation")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS / "A_seuil_dev.png", dpi=150)
plt.show()

# %% CELL 30
# Partie : A
# 7 — Choisir le seuil sur dev
seuils_a_tester = sorted(set([
    0.10, seuil_A, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70
]))

comparaison_seuils = []

for seuil in seuils_a_tester:
    predictions = (scores_val_A >= seuil).astype(int)

    comparaison_seuils.append({
        "seuil": seuil,
        "precision_yes": precision_score(
            y_val, predictions, zero_division=0
        ),
        "rappel_yes": recall_score(
            y_val, predictions, zero_division=0
        ),
        "f1_yes": f1_score(
            y_val, predictions, zero_division=0
        ),
        "macro_f1": f1_score(
            y_val, predictions,
            average="macro", zero_division=0
        ),
        "pourcentage_predit_Yes": 100 * predictions.mean()
    })

comparaison_seuils = pd.DataFrame(comparaison_seuils)

display(comparaison_seuils.round(4))

comparaison_seuils.to_csv(
    RESULTS / "A_comparaison_seuils_dev.csv",
    index=False,
    encoding="utf-8-sig"
)

# %% CELL 32
# Partie : A
# 8 — Afficher la comparaison finale et sauvegarder les prédictions
journal_A = pd.read_csv(RESULTS / "results.csv")

comparaison_A = (
    journal_A[
        journal_A["partie"].eq("A")
        & journal_A["split"].eq("dev")
        & journal_A["seed"].eq(SEED)
    ]
    .sort_values("f1_yes", ascending=False)
)

display(comparaison_A[[
    "modele", "seuil",
    "precision_yes", "rappel_yes",
    "f1_yes", "pr_auc", "macro_f1",
    "temps_entrainement_s"
]].round(4))

comparaison_A.to_csv(
    RESULTS / "A_comparaison_dev.csv",
    index=False,
    encoding="utf-8-sig"
)

predictions_A = val_A[[
    "tweet_id", "tweet_text", "class_label"
]].copy()

predictions_A["score_Yes"] = scores_val_A
predictions_A["prediction"] = np.where(
    pred_val_A == 1, "Yes", "No"
)
predictions_A["seuil"] = seuil_A

predictions_A.to_csv(
    RESULTS / "A_predictions_dev.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Résultats enregistrés dans :", RESULTS)

# %% CELL 37
# Partie : B
# 1 — Préparer l’évaluation et la sauvegarde
import json
import gc
import re
import torch
import numpy as np
import pandas as pd

from pathlib import Path
from time import perf_counter
from tqdm.auto import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline
)

from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    accuracy_score, precision_recall_curve,
    average_precision_score, auc
)

RESULTS = Path("/content/results")
RESULTS.mkdir(parents=True, exist_ok=True)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Même ordre que le dataframe officiel dev.
textes_dev = val["tweet_text"].tolist()
y_dev = val["class_label"].map({"No": 0, "Yes": 1}).to_numpy()

print("Appareil :", DEVICE)
print("Tweets dev :", len(textes_dev))


def choisir_seuil_B(y, scores):
    precision, rappel, seuils = precision_recall_curve(y, scores)

    p = precision[:-1]
    r = rappel[:-1]

    f1 = np.divide(
        2 * p * r,
        p + r,
        out=np.zeros_like(p),
        where=(p + r) != 0
    )

    candidats = np.flatnonzero(np.isclose(f1, f1.max()))
    indice = candidats[
        np.argmin(np.abs(seuils[candidats] - 0.5))
    ]

    return float(seuils[indice])


def evaluer_B(partie, nom, predictions, scores,
              seuil=0.5, temps_inference=0.0, notes=""):

    predictions = np.asarray(predictions, dtype=int)
    scores = np.asarray(scores, dtype=float)

    p, r, _ = precision_recall_curve(y_dev, scores)

    ligne = {
        "partie": partie,
        "modele": nom,
        "split": "dev",
        "seed": SEED,
        "seuil": seuil,
        "precision_yes": precision_score(
            y_dev, predictions, zero_division=0
        ),
        "rappel_yes": recall_score(
            y_dev, predictions, zero_division=0
        ),
        "f1_yes": f1_score(
            y_dev, predictions, zero_division=0
        ),
        "pr_auc": auc(r, p),
        "average_precision": average_precision_score(y_dev, scores),
        "macro_f1": f1_score(
            y_dev, predictions, average="macro", zero_division=0
        ),
        "accuracy": accuracy_score(y_dev, predictions),
        "temps_entrainement_s": 0.0,
        "temps_inference_s": temps_inference,
        "pourcentage_predit_Yes": 100 * predictions.mean(),
        "notes": notes
    }

    chemin = RESULTS / "results.csv"
    historique = (
        pd.read_csv(chemin)
        if chemin.exists()
        else pd.DataFrame()
    )

    # Remplacer la même expérience en cas de réexécution.
    if not historique.empty:
        masque = (
            historique["partie"].eq(partie)
            & historique["modele"].eq(nom)
            & historique["split"].eq("dev")
            & historique["seed"].eq(SEED)
        )
        historique = historique.loc[~masque]

    nouveau = pd.DataFrame([ligne])
    historique = (
        nouveau if historique.empty
        else pd.concat([historique, nouveau], ignore_index=True)
    )

    historique.to_csv(
        chemin, index=False, encoding="utf-8-sig"
    )

    display(nouveau.round(4))
    return ligne

# %% CELL 43
# Partie : B
# 4 — Calculer les scores NLI
@torch.inference_mode()
def scores_nli(textes, hypotheses, batch_size=8):
    scores = []

    for debut in tqdm(
        range(0, len(textes), batch_size),
        desc="Inférence NLI"
    ):
        batch = textes[debut:debut + batch_size]

        premises = []
        hypotheses_batch = []

        # Ordre fixe : Yes, puis No pour chaque tweet.
        for texte in batch:
            premises.extend([texte, texte])
            hypotheses_batch.extend([
                hypotheses["Yes"],
                hypotheses["No"]
            ])

        inputs = tokenizer_nli(
            premises,
            hypotheses_batch,
            padding=True,
            truncation="only_first",
            max_length=512,
            return_tensors="pt"
        ).to(DEVICE)

        logits = model_nli(**inputs).logits

        logits_entailment = logits[:, ENTAILMENT_ID].reshape(-1, 2)

        score_yes = torch.softmax(
            logits_entailment, dim=1
        )[:, 0]

        scores.extend(score_yes.cpu().numpy().tolist())

    return np.asarray(scores)

# %% CELL 45
# Partie : B
# 5 — Évaluer les trois formulations et choisir leurs seuils
predictions_B1 = val[[
    "tweet_id", "tweet_text", "class_label"
]].copy()

resultats_B1 = []
seuils_B1 = {}

for langue, hypotheses in HYPOTHESES.items():
    print(f"\nFormulation : {langue}")

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    debut = perf_counter()
    scores = scores_nli(textes_dev, hypotheses)

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    duree = perf_counter() - debut

    seuil_optimal = choisir_seuil_B(y_dev, scores)
    seuils_B1[langue] = seuil_optimal

    predictions_B1[f"score_Yes_{langue}"] = scores
    predictions_B1[f"prediction_{langue}"] = np.where(
        scores >= seuil_optimal, "Yes", "No"
    )

    for variante, seuil in [
        ("050", 0.5),
        ("optimise", seuil_optimal)
    ]:
        ligne = evaluer_B(
            partie="B1",
            nom=f"NLI_{langue}_{variante}",
            predictions=(scores >= seuil).astype(int),
            scores=scores,
            seuil=seuil,
            temps_inference=duree,
            notes=(
                f"Hypothèses {langue} ; aucun entraînement CheckThat ; "
                "tweets tronqués si la paire dépasse 512 tokens."
            )
        )
        resultats_B1.append(ligne)

    # Sauvegarde après chaque langue.
    predictions_B1.to_csv(
        RESULTS / "B1_predictions_dev.csv",
        index=False,
        encoding="utf-8-sig"
    )

comparaison_B1 = pd.DataFrame(resultats_B1)

comparaison_B1.to_csv(
    RESULTS / "B1_comparaison_dev.csv",
    index=False,
    encoding="utf-8-sig"
)

with open(
    RESULTS / "B1_seuils.json", "w", encoding="utf-8"
) as f:
    json.dump(seuils_B1, f, indent=2)

display(comparaison_B1[[
    "modele", "seuil", "precision_yes",
    "rappel_yes", "f1_yes", "pr_auc", "macro_f1"
]].round(4))

# %% CELL 49
# Partie : B
# 7 — Appliquer la règle « entité + nombre → Yes »
def contient_nombre_B(text):
    text = re.sub(PATTERNS["url"], " ", text)
    text = re.sub(PATTERNS["mention"], " ", text)
    return bool(re.search(PATTERNS["nombre"], text))


if torch.cuda.is_available():
    torch.cuda.synchronize()

debut = perf_counter()

entites_dev = []

for start in tqdm(
    range(0, len(textes_dev), 16),
    desc="Détection des entités"
):
    batch = textes_dev[start:start + 16]

    entites_dev.extend(
        ner(batch, batch_size=8)
    )

if torch.cuda.is_available():
    torch.cuda.synchronize()

temps_ner = perf_counter() - debut

predictions_B2 = val[[
    "tweet_id", "tweet_text", "class_label"
]].copy()

predictions_B2["nb_entites"] = [
    len(entites) for entites in entites_dev
]

predictions_B2["has_entite"] = (
    predictions_B2["nb_entites"] > 0
)

predictions_B2["has_nombre"] = (
    predictions_B2["tweet_text"].map(contient_nombre_B)
)

# Au moins une entité ET au moins un nombre.
pred_b2 = (
    predictions_B2["has_entite"]
    & predictions_B2["has_nombre"]
).astype(int).to_numpy()

predictions_B2["prediction"] = np.where(
    pred_b2 == 1, "Yes", "No"
)

# Conserver les entités avec leurs positions dans le texte.
predictions_B2["entites"] = [
    json.dumps([
        {
            "texte": texte[int(e["start"]):int(e["end"])],
            "type": e["entity_group"],
            "score": float(e["score"])
        }
        for e in entites
    ], ensure_ascii=False)
    for texte, entites in zip(textes_dev, entites_dev)
]

evaluer_B(
    partie="B2",
    nom="CAMeLBERT_entite_et_nombre",
    predictions=pred_b2,
    scores=pred_b2.astype(float),
    seuil=0.5,
    temps_inference=temps_ner,
    notes=(
        "Yes si au moins une entité NER et un nombre hors URL/mentions ; "
        "score binaire ; aucun entraînement CheckThat."
    )
)

predictions_B2.to_csv(
    RESULTS / "B2_predictions_dev.csv",
    index=False,
    encoding="utf-8-sig"
)

display(predictions_B2[[
    "tweet_text", "class_label", "nb_entites",
    "has_nombre", "prediction", "entites"
]].head(10))

# %% CELL 51
# Partie : B
# 8 — Comparer les résultats de B
journal = pd.read_csv(RESULTS / "results.csv")

comparaison_B = journal[
    journal["partie"].isin(["B1", "B2"])
    & journal["split"].eq("dev")
    & journal["seed"].eq(SEED)
].sort_values("f1_yes", ascending=False)

display(comparaison_B[[
    "partie", "modele", "seuil",
    "precision_yes", "rappel_yes",
    "f1_yes", "pr_auc", "average_precision", "macro_f1"
]].round(4))

comparaison_B.to_csv(
    RESULTS / "B_comparaison_dev.csv",
    index=False,
    encoding="utf-8-sig"
)

# %% CELL 69
# Partie : C
# 8 — Visualiser l'attention de C2 sur cinq exemples de dev
import matplotlib.pyplot as plt

# Sélectionner et recharger le meilleur C2
meilleur_c2 = max(
    [e for e in experiences_C if e["arch"] == "C2"],
    key=lambda e: e["f1"]
)

matrice = (
    ALEATOIRE if meilleur_c2["embedding"] == "random"
    else ARAVEC
)

c2 = ModeleMotsC("C2", matrice).to(DEVICE)

c2.load_state_dict(
    torch.load(
        meilleur_c2["checkpoint"],
        map_location=DEVICE,
        weights_only=True
    )
)
c2.eval()

print("Modèle visualisé :", meilleur_c2["nom"])

attention_records = []

for i in range(min(5, len(val_C))):
    with torch.inference_mode():
        _, poids = c2(
            VM[i:i+1].to(DEVICE),
            return_attention=True
        )

    tokens = vw[i][:LM] or ["<UNK>"]
    valeurs = poids[0, :len(tokens)].cpu().numpy()

    tableau = pd.DataFrame({
        "position": range(len(tokens)),
        "token_original": tokens,
        "token_modele": [
            t if t in vocab_mots else "<UNK>"
            for t in tokens
        ],
        "attention": valeurs
    })

    print("\nTweet :", val_C.iloc[i]["tweet_text"])
    print("Label réel :", val_C.iloc[i]["class_label"])

    # Afficher les 10 tokens les plus pondérés
    display(
        tableau.sort_values(
            "attention", ascending=False
        ).head(10)
    )

    # Graphique des poids de tous les tokens
    plt.figure(figsize=(10, 2.5))
    plt.bar(tableau["position"], tableau["attention"])
    plt.xlabel("Position du token")
    plt.ylabel("Poids d’attention")
    plt.title(f"C2 — exemple dev {i}")
    plt.tight_layout()
    plt.savefig(OUT_C / f"attention_dev_{i}.png", dpi=150)
    plt.show()
    plt.close()

    tableau["tweet_id"] = val_C.iloc[i]["tweet_id"]
    attention_records.append(tableau)

pd.concat(attention_records, ignore_index=True).to_csv(
    OUT_C / "attention_C2.csv",
    index=False,
    encoding="utf-8-sig"
)

del c2

# %% CELL 75
# Partie : D
# 2. Extraire les vecteurs pour D1 et D2
@torch.inference_mode()
def extraire_D(textes):
    textes = [
        preprocessor_d.preprocess(t) or tokenizer_d.unk_token
        for t in textes
    ]

    n = len(textes)
    dim = arabert_d.config.hidden_size

    cls = np.zeros((n, dim), dtype=np.float32)
    moyenne = np.zeros_like(cls)
    etats = torch.zeros(n, MAX_LEN_D - 2, dim)
    longueurs = torch.zeros(n, dtype=torch.long)
    tokens = []

    for debut in tqdm(range(0, n, 16)):
        inputs = tokenizer_d(
            textes[debut:debut + 16],
            padding=True,
            truncation=True,
            max_length=MAX_LEN_D,
            return_special_tokens_mask=True,
            return_tensors="pt"
        )

        speciaux = inputs.pop("special_tokens_mask").bool()
        masque = inputs["attention_mask"].bool() & ~speciaux

        h = arabert_d(**{
            k: v.to(DEVICE) for k, v in inputs.items()
        }).last_hidden_state.cpu()

        for j in range(len(h)):
            i = debut + j
            vecteurs = h[j, masque[j]]
            ids = inputs["input_ids"][j, masque[j]]

            if len(vecteurs) == 0:
                raise ValueError(f"Tweet sans token de contenu : {i}")

            longueur = len(vecteurs)

            cls[i] = h[j, 0].numpy()
            moyenne[i] = vecteurs.mean(0).numpy()
            etats[i, :longueur] = vecteurs
            longueurs[i] = longueur
            tokens.append(tokenizer_d.convert_ids_to_tokens(ids.tolist()))

    return cls, moyenne, etats, longueurs, tokens


cls_train, mean_train, H_train, L_train, _ = extraire_D(
    train_fit["tweet_text"].tolist()
)

cls_dev, mean_dev, H_dev, L_dev, tokens_D = extraire_D(
    val["tweet_text"].tolist()
)

del arabert_d
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# %% CELL 77
# Partie : D
# 3.EVALUATION ET SAUVGARDE
def threshold_and_metrics(y, scores, threshold=None):
    p, r, seuils = precision_recall_curve(y, scores)

    if threshold is None:
        f1 = np.divide(
            2 * p[:-1] * r[:-1],
            p[:-1] + r[:-1],
            out=np.zeros_like(p[:-1]),
            where=(p[:-1] + r[:-1]) != 0
        )

        candidats = np.flatnonzero(np.isclose(f1, f1.max()))
        indice = candidats[
            np.argmin(np.abs(seuils[candidats] - 0.5))
        ]
        threshold = float(seuils[indice])

    predictions = (np.asarray(scores) >= threshold).astype(int)

    return {
        "seuil": threshold,
        "precision_yes": precision_score(
            y, predictions, zero_division=0
        ),
        "rappel_yes": recall_score(
            y, predictions, zero_division=0
        ),
        "f1_yes": f1_score(
            y, predictions, zero_division=0
        ),
        "macro_f1": f1_score(
            y, predictions, average="macro", zero_division=0
        ),
        "pr_auc": auc(r, p)
    }

# %% CELL 78
# Partie : D
# 3.EVALUATION ET SAUVGARDE
seuils_D = {}

def enregistrer_D(nom, scores, duree):
    lignes = []

    for variante, seuil in [("050", 0.5), ("optimise", None)]:
        mesures = threshold_and_metrics(ydev_d, scores, seuil)

        lignes.append({
            "partie": "D",
            "modele": f"{nom}_{variante}",
            "split": "dev",
            "seed": SEED_D,
            "temps_entrainement_s": duree,
            **mesures
        })

    seuils_D[nom] = lignes[1]["seuil"]

    chemin = RESULTS / "results.csv"
    historique = (
        pd.read_csv(chemin)
        if chemin.exists()
        else pd.DataFrame()
    )

    if not historique.empty:
        remplacer = (
            historique["partie"].eq("D")
            & historique["modele"].isin([r["modele"] for r in lignes])
            & historique["split"].eq("dev")
            & historique["seed"].eq(SEED_D)
        )
        historique = historique.loc[~remplacer]

    nouveaux = pd.DataFrame(lignes)
    pd.concat([historique, nouveaux], ignore_index=True).to_csv(
        chemin, index=False, encoding="utf-8-sig"
    )

    display(nouveaux.round(4))

    (OUT_D / "seuils.json").write_text(
        json.dumps(seuils_D, indent=2), encoding="utf-8"
    )

# %% CELL 80
# Partie : D
# 4. D1 — CLS vs moyenne → régression logistique
for nom, X_train_D, X_dev_D in [
    ("D1_CLS", cls_train, cls_dev),
    ("D1_moyenne", mean_train, mean_dev)
]:
    modele_lr = LogisticRegression(
        class_weight="balanced",
        solver="liblinear",
        max_iter=2000,
        random_state=SEED_D
    )

    debut = perf_counter()
    modele_lr.fit(X_train_D, ytrain_d)
    duree = perf_counter() - debut

    colonne_yes = list(modele_lr.classes_).index(1)
    scores = modele_lr.predict_proba(X_dev_D)[:, colonne_yes]

    enregistrer_D(nom, scores, duree)
    joblib.dump(modele_lr, OUT_D / f"{nom}.joblib")

# %% CELL 86
# Partie : D
# 7. Visualiser l’attention sur cinq tweets de dev
modele_D2.eval()
attention_D = []

for i in range(min(5, len(val))):
    with torch.inference_mode():
        _, poids = modele_D2(
            H_dev[i:i+1].to(DEVICE),
            L_dev[i:i+1],
            return_attention=True
        )

    valeurs = poids[0, :L_dev[i].item()].cpu().numpy()

    tableau = pd.DataFrame({
        "position": range(len(tokens_D[i])),
        "token": tokens_D[i],
        "attention": valeurs
    })

    print("\nTweet :", val.iloc[i]["tweet_text"])
    display(tableau.sort_values("attention", ascending=False).head(10))

    plt.figure(figsize=(10, 2.5))
    plt.bar(tableau["position"], tableau["attention"])
    plt.xlabel("Position du token")
    plt.ylabel("Attention")
    plt.title(f"D2 — exemple dev {i}")
    plt.tight_layout()
    plt.savefig(OUT_D / f"D2_attention_{i}.png", dpi=150)
    plt.show()

    tableau["tweet_id"] = val.iloc[i]["tweet_id"]
    attention_D.append(tableau)

pd.concat(attention_D, ignore_index=True).to_csv(
    OUT_D / "D2_attention.csv",
    index=False,
    encoding="utf-8-sig"
)

# %% CELL 91
# Partie : E
# 2 — Imports nécessaires
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_linear_schedule_with_warmup
)

from arabert.preprocess import ArabertPreprocessor

# %% CELL 103
# Partie : E
# 8 — Calculer les scores sur dev
@torch.inference_mode()
def predire_E(model, tokens, langue="ar"):
    model.eval()
    scores = []

    for debut in range(
        0,
        len(tokens["input_ids"]),
        CFG_E["micro_batch"]
    ):
        fin = debut + CFG_E["micro_batch"]

        inputs = {
            nom: valeurs[debut:fin].to(DEVICE_E)
            for nom, valeurs in tokens.items()
        }

        with torch.autocast(
            device_type=DEVICE_E.type,
            dtype=torch.float16,
            enabled=DEVICE_E.type == "cuda"
        ):
            logits = model(inputs, langue)

        scores.extend(
            logits.float().sigmoid().cpu().tolist()
        )

    return np.array(scores)

# %% CELL 111
# Partie : E
# 12 — Sauvegarder les résultats et la configuration
def sauvegarder_E(
    nom, partie, model_id, model, tokenizer,
    train_tokens, scores, seed, lambda_en,
    duree, meilleure_epoque, folder
):
    lignes = []

    for variante, seuil in [
        ("050", 0.5),
        ("optimise", None)
    ]:
        lignes.append({
            "partie": partie,
            "modele": f"{nom}_{variante}",
            "split": "dev",
            "seed": seed,
            "lambda_en": lambda_en,
            "temps_entrainement_s": duree,
            "meilleure_epoque": meilleure_epoque,
            "parametres_entrainables": sum(
                p.numel()
                for p in model.parameters()
                if p.requires_grad
            ),
            **threshold_and_metrics(
                Y_DEV_E, scores, seuil
            )
        })

    chemin = RESULTS / "results.csv"

    ancien = (
        pd.read_csv(chemin)
        if chemin.exists()
        else pd.DataFrame()
    )

    nouveaux = pd.DataFrame(lignes)

    if not ancien.empty:
        retirer = (
            ancien["partie"].eq(partie)
            & ancien["modele"].isin(nouveaux["modele"])
            & ancien["seed"].eq(seed)
            & ancien["split"].eq("dev")
        )
        ancien = ancien.loc[~retirer]

    pd.concat(
        [ancien, nouveaux], ignore_index=True
    ).to_csv(
        chemin, index=False, encoding="utf-8-sig"
    )

    nouveaux.to_csv(
        folder / "resultats_dev.csv", index=False
    )

    predictions = AR_DEV[
        ["tweet_id", "tweet_text", "class_label"]
    ].copy()

    predictions["score_Yes"] = scores
    predictions["prediction"] = np.where(
        scores >= lignes[1]["seuil"], "Yes", "No"
    )

    predictions.to_csv(
        folder / "predictions_dev.csv",
        index=False,
        encoding="utf-8-sig"
    )

    tokenizer.save_pretrained(folder / "tokenizer")

    configuration = {
        "model_id": model_id,
        "deux_tetes": partie == "E2",
        "max_length": train_tokens["input_ids"].shape[1],
        "seed": seed,
        "lambda_en": lambda_en,
        "seuil": lignes[1]["seuil"],
        "config": CFG_E,
        "labels": {"No": 0, "Yes": 1},
        "revision": getattr(
            model.encodeur.config, "_commit_hash", None
        ),
        "preprocessing": (
            "ArabertPreprocessor"
            if "arabert" in model_id
            else "texte_original"
        )
    }

    (folder / "configuration.json").write_text(
        json.dumps(configuration, indent=2),
        encoding="utf-8"
    )

    return nouveaux

# %% CELL 121
# Partie : E
# 17 — Afficher les résultats de chaque graine
table_E = pd.concat(
    RESULTATS_E,
    ignore_index=True
)

colonnes = [
    "partie", "modele", "seed", "seuil",
    "precision_yes", "rappel_yes",
    "f1_yes", "pr_auc", "macro_f1"
]

display(table_E[colonnes].round(4))

table_E.to_csv(
    OUT_E / "comparaison_par_graine.csv",
    index=False,
    encoding="utf-8-sig"
)

# %% CELL 123
# Partie : E
# 18 — Comparer les moyennes des trois graines
optimises = table_E[
    table_E["modele"].str.endswith("_optimise")
]

resume_E = optimises.groupby(
    ["partie", "modele", "lambda_en"],
    as_index=False
).agg(
    graines=("seed", "nunique"),
    f1_moyen=("f1_yes", "mean"),
    f1_ecart_type=("f1_yes", "std"),
    pr_auc_moyen=("pr_auc", "mean"),
    macro_f1_moyen=("macro_f1", "mean"),
    temps_moyen_s=("temps_entrainement_s", "mean")
)

resume_E = resume_E.sort_values(
    "f1_moyen",
    ascending=False
)

display(resume_E.round(4))

resume_E.to_csv(
    OUT_E / "moyenne_3_graines.csv",
    index=False,
    encoding="utf-8-sig"
)

choix_lambda = resume_E[
    resume_E["partie"].eq("E2")
].iloc[0]

print(
    "Lambda retenu sur dev arabe :",
    choix_lambda["lambda_en"]
)

# %% CELL 124
# Partie : E
# 18 — Comparer les moyennes des trois graines
import matplotlib.pyplot as plt
import numpy as np

DOSSIER_PLOTS = OUT_E / "plots"
DOSSIER_PLOTS.mkdir(exist_ok=True)

assert resume_E["graines"].eq(3).all(), (
    "Terminer les trois graines de chaque expérience avant ces graphiques."
)

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False
})


def sauvegarder_plot(fig, nom):
    fig.text(
        0.5, 0.02,
        "Dev arabe • seuil choisi sur dev • moyenne ± écart-type, 3 graines",
        ha="center",
        fontsize=9,
        color="dimgray"
    )

    fig.tight_layout(rect=[0, 0.07, 1, 1])

    fig.savefig(
        DOSSIER_PLOTS / f"{nom}.png",
        dpi=300,
        bbox_inches="tight"
    )
    fig.savefig(
        DOSSIER_PLOTS / f"{nom}.pdf",
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

# %% CELL 126
# Partie : E
# Graphique 1 — Comparaison des deux modèles fine-tunés
e1 = (
    resume_E[resume_E["partie"].eq("E1")]
    .sort_values("modele")
    .copy()
)

noms = {
    "E1_AraBERT_optimise": "AraBERT Twitter",
    "E1_MARBERTv2_optimise": "MARBERTv2"
}

labels = e1["modele"].map(noms).fillna(e1["modele"])
moyennes = e1["f1_moyen"].to_numpy()
ecarts = e1["f1_ecart_type"].to_numpy()
positions = np.arange(len(e1))

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(
    positions,
    moyennes,
    yerr=ecarts,
    capsize=7,
    width=0.55,
    color=["#2563EB", "#059669"],
    error_kw={"elinewidth": 1.5}
)

for x, moyenne, ecart in zip(positions, moyennes, ecarts):
    ax.annotate(
        f"{moyenne:.4f}",
        xy=(x, moyenne + ecart),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontweight="bold"
    )

ax.set_xticks(positions, labels)
ax.set_ylabel("F1 de la classe Yes")
ax.set_ylim(0, 1)
ax.set_title("E1 — Comparaison des modèles fine-tunés")
ax.yaxis.grid(True, alpha=0.2)
ax.set_axisbelow(True)

sauvegarder_plot(fig, "E1_comparaison_modeles")

# %% CELL 128
# Partie : E
# E2 — Effet de l’apprentissage avec l’anglais
e2 = (
    resume_E[resume_E["partie"].eq("E2")]
    .sort_values("lambda_en")
    .copy()
)

lambdas = e2["lambda_en"].to_numpy()
moyennes = e2["f1_moyen"].to_numpy()
ecarts = e2["f1_ecart_type"].to_numpy()

reference = e2.loc[
    e2["lambda_en"].eq(0), "f1_moyen"
].iloc[0]

fig, ax = plt.subplots(figsize=(8, 5))

ax.errorbar(
    lambdas,
    moyennes,
    yerr=ecarts,
    fmt="-o",
    color="#7C3AED",
    linewidth=2,
    markersize=8,
    capsize=6,
    label="XLM-R : moyenne ± écart-type"
)

ax.axhline(
    reference,
    color="gray",
    linestyle="--",
    label="Référence arabe seul : λ = 0"
)

for x, moyenne, ecart in zip(lambdas, moyennes, ecarts):
    ax.annotate(
        f"{moyenne:.4f}",
        xy=(x, moyenne + ecart),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center"
    )

ax.set_xticks(lambdas)
ax.set_xlabel("λ — coefficient de la perte anglaise")
ax.set_ylabel("F1 de la classe Yes")
ax.set_ylim(0, 1)
ax.set_title("E2 — Effet de l’apprentissage avec l’anglais")
ax.grid(True, alpha=0.2)
ax.legend(loc="best", fontsize=10)

sauvegarder_plot(fig, "E2_effet_lambda")

print("Graphiques sauvegardés dans :", DOSSIER_PLOTS)

# %% CELL 132
# Partie : F1
# F1.1 — Retenir le modèle à partir de dev
import matplotlib.pyplot as plt

OUT_F = RESULTS / "F"
OUT_F.mkdir(exist_ok=True)

selection = resume_E.sort_values(
    "f1_moyen", ascending=False
).iloc[0]

assert selection["graines"] == 3, (
    "Terminer les trois graines du modèle avant F1."
)

MODELE_F1 = selection["modele"].removesuffix("_optimise")

print("Modèle retenu sur dev :", MODELE_F1)

# Conserver les seuils appris sur dev, séparément pour chaque graine.
CONFIGS_F1 = {}

for seed in SEEDS_E:
    dossier = OUT_E / f"{MODELE_F1}_seed{seed}"

    CONFIGS_F1[seed] = json.loads(
        (dossier / "configuration.json").read_text(
            encoding="utf-8"
        )
    )

    print(
        "Graine :", seed,
        "| seuil dev :", CONFIGS_F1[seed]["seuil"]
    )

(OUT_F / "F1_choix_sur_dev.json").write_text(
    json.dumps({
        "modele": MODELE_F1,
        "seuils": {
            str(seed): config["seuil"]
            for seed, config in CONFIGS_F1.items()
        }
    }, indent=2),
    encoding="utf-8"
)

# %% CELL 136
# Partie : F1
# F1.3 — Appliquer les seuils de dev et tracer les courbes
resultats_F1 = []
comparaison_seuils_F1 = []

for seed in SEEDS_E:
    dossier = OUT_E / f"{MODELE_F1}_seed{seed}"
    config = CONFIGS_F1[seed]
    seuil_dev = config["seuil"]

    tokenizer = AutoTokenizer.from_pretrained(
        dossier / "tokenizer"
    )

    model = ModeleE(
        config["model_id"],
        deux_tetes=config["deux_tetes"]
    ).to(DEVICE_E)

    model.load_state_dict(
        torch.load(
            dossier / "model.pt",
            map_location=DEVICE_E,
            weights_only=True
        )
    )

    prep = (
        ArabertPreprocessor(model_name=config["model_id"])
        if config["preprocessing"] == "ArabertPreprocessor"
        else None
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for ax, (split, df) in zip(
        axes,
        [("dev-test", dev_test), ("test", test)]
    ):
        textes = [
            prep.preprocess(t) if prep else str(t)
            for t in df["tweet_text"]
        ]

        textes = [
            t.strip() or tokenizer.unk_token
            for t in textes
        ]

        tokens = tokenizer(
            textes,
            padding="max_length",
            truncation=True,
            max_length=config["max_length"],
            return_tensors="pt"
        )

        y = df["class_label"].map(
            {"No": 0, "Yes": 1}
        ).to_numpy()

        scores = predire_E(model, tokens)

        # Évaluation principale : seuil de dev inchangé.
        mesures = threshold_and_metrics(
            y, scores, seuil_dev
        )

        resultats_F1.append({
            "partie": "F1",
            "modele": MODELE_F1,
            "split": split,
            "seed": seed,
            **mesures
        })

        # Analyse a posteriori uniquement.
        posteriori = threshold_and_metrics(y, scores)

        comparaison_seuils_F1.append({
            "modele": MODELE_F1,
            "split": split,
            "seed": seed,
            "seuil_dev": seuil_dev,
            "seuil_optimal_posteriori": posteriori["seuil"],
            "ecart_seuil": posteriori["seuil"] - seuil_dev,
            "f1_seuil_dev": mesures["f1_yes"],
            "f1_optimal_posteriori": posteriori["f1_yes"]
        })

        # Courbe comprenant les seuils où les prédictions changent.
        seuils = np.unique(np.concatenate([
            [0.0, 1.0, seuil_dev],
            scores
        ]))

        f1_seuils = [
            f1_score(
                y, scores >= seuil, zero_division=0
            )
            for seuil in seuils
        ]

        pd.DataFrame({
            "seuil": seuils,
            "f1_yes": f1_seuils
        }).to_csv(
            OUT_F / f"F1_courbe_{split}_seed{seed}.csv",
            index=False
        )

        ax.plot(seuils, f1_seuils, label="F1-Yes")

        ax.axvline(
            seuil_dev,
            color="red",
            linestyle="--",
            label=f"Seuil dev : {seuil_dev:.3f}"
        )

        ax.scatter(
            posteriori["seuil"],
            posteriori["f1_yes"],
            color="green",
            label="Optimum a posteriori",
            zorder=3
        )

        ax.set(
            title=f"{split} — graine {seed}",
            xlabel="Seuil",
            ylabel="F1 de Yes",
            xlim=(0, 1),
            ylim=(0, 1)
        )

        ax.grid(alpha=0.2)
        ax.legend(fontsize=9)

        # Conserver les scores et décisions officielles.
        predictions = df[
            ["tweet_id", "tweet_text", "class_label"]
        ].copy()

        predictions["score_Yes"] = scores
        predictions["seuil_dev"] = seuil_dev
        predictions["prediction"] = np.where(
            scores >= seuil_dev, "Yes", "No"
        )

        predictions.to_csv(
            OUT_F / f"F1_predictions_{split}_seed{seed}.csv",
            index=False,
            encoding="utf-8-sig"
        )

    fig.suptitle(MODELE_F1)
    fig.tight_layout()

    fig.savefig(
        OUT_F / f"F1_courbes_seed{seed}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()
    plt.close(fig)

    del model, tokenizer
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %% CELL 138
# Partie : F1
# F1.4 — Afficher et sauvegarder les résultats
table_F1 = pd.DataFrame(resultats_F1)
seuils_F1 = pd.DataFrame(comparaison_seuils_F1)

display(table_F1.round(4))
display(seuils_F1.round(4))

table_F1.to_csv(
    OUT_F / "F1_resultats.csv",
    index=False,
    encoding="utf-8-sig"
)

seuils_F1.to_csv(
    OUT_F / "F1_seuils_posteriori.csv",
    index=False,
    encoding="utf-8-sig"
)

# Ajouter F1 au journal, sans dupliquer les mêmes expériences.
chemin = RESULTS / "results.csv"
journal = pd.read_csv(chemin)

remplacer = (
    journal["partie"].eq("F1")
    & journal["modele"].eq(MODELE_F1)
    & journal["seed"].isin(SEEDS_E)
    & journal["split"].isin(["dev-test", "test"])
)

journal = pd.concat(
    [journal.loc[~remplacer], table_F1],
    ignore_index=True
)

journal.to_csv(
    chemin,
    index=False,
    encoding="utf-8-sig"
)

print("Résultats et graphiques sauvegardés dans :", OUT_F)

# %% CELL 141
# Partie : F2
# F2.1 — Comparer les scores sur les trois graines
comparaison_F2 = table_F1.groupby("split").agg(
    f1_moyen=("f1_yes", "mean"),
    f1_ecart_type=("f1_yes", "std"),
    precision_moyenne=("precision_yes", "mean"),
    rappel_moyen=("rappel_yes", "mean"),
    pr_auc_moyen=("pr_auc", "mean"),
    macro_f1_moyen=("macro_f1", "mean")
).reset_index()

display(comparaison_F2.round(4))

comparaison_F2.to_csv(
    OUT_F / "F2_comparaison_scores.csv",
    index=False,
    encoding="utf-8-sig"
)

# Graphique pour la présentation
fig, ax = plt.subplots(figsize=(7, 4))

ax.bar(
    comparaison_F2["split"],
    comparaison_F2["f1_moyen"],
    yerr=comparaison_F2["f1_ecart_type"],
    capsize=6,
    color=["#2563EB", "#059669"]
)

ax.set(
    ylabel="F1 de Yes",
    title="F2 — Dev-test vs test",
    ylim=(0, 1)
)

ax.yaxis.grid(True, alpha=0.2)
ax.set_axisbelow(True)

fig.text(
    0.5, 0.01,
    "Seuils de dev inchangés — moyenne ± écart-type sur 3 graines",
    ha="center",
    fontsize=9
)

fig.tight_layout(rect=[0, 0.06, 1, 1])
fig.savefig(
    OUT_F / "F2_comparaison_scores.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# %% CELL 143
# Partie : F2
# F2.2 — Comparer les types d’erreurs
erreurs_F2 = []
bilan_F2 = []

for seed in SEEDS_E:
    for split in ["dev-test", "test"]:
        df = pd.read_csv(
            OUT_F / f"F1_predictions_{split}_seed{seed}.csv",
            dtype={"tweet_id": str}
        )

        vrai_yes = df["class_label"].eq("Yes")
        predit_yes = df["prediction"].eq("Yes")

        fp = ~vrai_yes & predit_yes
        fn = vrai_yes & ~predit_yes

        bilan_F2.append({
            "split": split,
            "seed": seed,
            "effectif": len(df),
            "pourcentage_Yes": 100 * vrai_yes.mean(),
            "FP": int(fp.sum()),
            "FN": int(fn.sum()),
            "taux_FP": fp.sum() / max((~vrai_yes).sum(), 1),
            "taux_FN": fn.sum() / max(vrai_yes.sum(), 1)
        })

        erreurs = df.loc[fp | fn].copy()
        erreurs["type_erreur"] = np.where(
            erreurs["class_label"].eq("No"), "FP", "FN"
        )
        erreurs["split"] = split
        erreurs["seed"] = seed

        erreurs_F2.append(erreurs)

erreurs_F2 = pd.concat(erreurs_F2, ignore_index=True)
bilan_F2 = pd.DataFrame(bilan_F2)

display(bilan_F2.round(4))

bilan_F2.to_csv(
    OUT_F / "F2_bilan_erreurs.csv",
    index=False,
    encoding="utf-8-sig"
)

erreurs_F2.to_csv(
    OUT_F / "F2_toutes_erreurs.csv",
    index=False,
    encoding="utf-8-sig"
)

# %% CELL 145
# Partie : F2
# F2.3 — Repérer les sujets présents dans les erreurs
from sklearn.feature_extraction.text import TfidfVectorizer

stop_ar = [
    normaliser_arabe(mot)
    for mot in [
        "في", "من", "على", "إلى", "عن", "مع",
        "هذا", "هذه", "ذلك", "التي", "الذي",
        "أن", "إن", "كان", "ما", "لا", "و", "يا"
    ]
]

def texte_themes_F2(texte):
    texte = re.sub(r"https?://\S+|www\.\S+", " ", str(texte))
    texte = re.sub(r"@\w+", " ", texte)
    return normaliser_arabe(texte)

vectorizer_F2 = TfidfVectorizer(
    preprocessor=texte_themes_F2,
    lowercase=False,
    token_pattern=r"(?u)\b[^\W\d_]{2,}\b",
    stop_words=stop_ar,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

vectorizer_F2.fit(train_fit["tweet_text"])
vocabulaire_F2 = vectorizer_F2.get_feature_names_out()

lecture_F2 = erreurs_F2[
    erreurs_F2["seed"].eq(42)
].copy()

mots_cles_F2 = []

for (split, type_erreur), groupe in lecture_F2.groupby(
    ["split", "type_erreur"]
):
    matrice = vectorizer_F2.transform(groupe["tweet_text"])
    importance = np.asarray(matrice.mean(axis=0)).ravel()

    indices = np.argsort(importance)[::-1][:15]

    for indice in indices:
        if importance[indice] > 0:
            mots_cles_F2.append({
                "split": split,
                "type_erreur": type_erreur,
                "expression": vocabulaire_F2[indice],
                "importance": importance[indice]
            })

mots_cles_F2 = pd.DataFrame(mots_cles_F2)

display(mots_cles_F2)

mots_cles_F2.to_csv(
    OUT_F / "F2_mots_cles_erreurs.csv",
    index=False,
    encoding="utf-8-sig"
)

# %% CELL 147
# Partie : F2
# F2.4 — Préparer la lecture des erreurs par thème
fichier_themes = OUT_F / f"F2_themes_{MODELE_F1}.csv"

if not fichier_themes.exists():
    lecture_F2["theme"] = ""
    lecture_F2["commentaire"] = ""

    lecture_F2.to_csv(
        fichier_themes,
        index=False,
        encoding="utf-8-sig"
    )

annotations_F2 = pd.read_csv(
    fichier_themes,
    dtype={"tweet_id": str},
    keep_default_na=False
)

pd.set_option("display.max_colwidth", 180)

display(annotations_F2[[
    "split", "type_erreur", "tweet_text",
    "class_label", "prediction", "theme"
]].head(30))

print("Fichier à compléter :", fichier_themes)

# %% CELL 149
# Partie : F2
# F2.5 — Résumer les thèmes confirmés
annotations_F2 = pd.read_csv(
    fichier_themes,
    dtype={"tweet_id": str},
    keep_default_na=False
)

annotees = annotations_F2[
    annotations_F2["theme"].str.strip().ne("")
].copy()

if annotees.empty:
    print("Lire les erreurs et compléter la colonne 'theme'.")
else:
    resume_themes_F2 = annotees.groupby(
        ["split", "type_erreur", "theme"]
    ).size().reset_index(name="nombre_erreurs")

    resume_themes_F2["pourcentage"] = (
        100 * resume_themes_F2["nombre_erreurs"]
        / resume_themes_F2.groupby(
            ["split", "type_erreur"]
        )["nombre_erreurs"].transform("sum")
    )

    display(
        resume_themes_F2.sort_values(
            ["split", "nombre_erreurs"],
            ascending=[True, False]
        ).round(2)
    )

    resume_themes_F2.to_csv(
        OUT_F / "F2_resume_themes.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print(
        "Erreurs annotées :",
        len(annotees), "/", len(annotations_F2)
    )

# %% CELL 154
# Partie : F3
# F3.2 — Calculer les scores par sous-groupe
def scores_groupes_F3(df, variante, split, seed):
    df = df.copy()

    df["has_nombre"] = df["tweet_text"].map(contient_nombre_F3)
    df["has_url"] = df["tweet_text"].map(contient_url_F3)

    groupes = {
        "tous": df,
        "avec_nombre": df[df["has_nombre"]],
        "sans_nombre": df[~df["has_nombre"]],
        "avec_URL": df[df["has_url"]],
        "sans_URL": df[~df["has_url"]]
    }

    lignes = []

    for groupe, sous_df in groupes.items():
        if sous_df.empty:
            continue

        y = sous_df["class_label"].eq("Yes").astype(int).to_numpy()
        pred = sous_df["prediction"].eq("Yes").astype(int).to_numpy()
        scores = sous_df["score_Yes"].to_numpy()

        pr_auc = np.nan
        if y.sum() > 0:
            p, r, _ = precision_recall_curve(y, scores)
            pr_auc = auc(r, p)

        lignes.append({
            "partie": "F3",
            "variante": variante,
            "split": split,
            "seed": seed,
            "groupe": groupe,
            "effectif": len(y),
            "Yes": int(y.sum()),
            "No": int(len(y) - y.sum()),
            "precision_yes": precision_score(y, pred, zero_division=0),
            "rappel_yes": recall_score(y, pred, zero_division=0),
            "f1_yes": f1_score(y, pred, zero_division=0),
            "macro_f1": f1_score(
                y, pred, labels=[0, 1],
                average="macro", zero_division=0
            ),
            "pr_auc": pr_auc
        })

    return lignes

# %% CELL 156
# Partie : F3
# F3.3 — Évaluer le modèle original
resultats_F3 = []

for seed in SEEDS_E:
    for split in ["dev-test", "test"]:
        predictions = pd.read_csv(
            OUT_F / f"F1_predictions_{split}_seed{seed}.csv",
            dtype={"tweet_id": str}
        )

        resultats_F3.extend(
            scores_groupes_F3(
                predictions, "original", split, seed
            )
        )

table_original_F3 = pd.DataFrame(resultats_F3)

display(
    table_original_F3.groupby(["split", "groupe"])[
        ["f1_yes", "precision_yes", "rappel_yes", "pr_auc"]
    ].mean().round(4)
)

table_original_F3.to_csv(
    OUT_F / "F3_scores_original.csv",
    index=False,
    encoding="utf-8-sig"
)

# %% CELL 162
# Partie : F3
# F3.6 — Évaluer le modèle réentraîné
for seed in SEEDS_E:
    dossier = OUT_E / f"{NOM_F3}_seed{seed}"

    configuration = json.loads(
        (dossier / "configuration.json").read_text()
    )

    modele = ModeleE(
        configuration["model_id"],
        deux_tetes=configuration["deux_tetes"]
    ).to(DEVICE_E)

    modele.load_state_dict(
        torch.load(
            dossier / "model.pt",
            map_location=DEVICE_E,
            weights_only=True
        )
    )

    for split, df in [("dev-test", dev_test), ("test", test)]:
        tokens = tokeniser_masque_F3(df["tweet_text"])
        scores = predire_E(modele, tokens)

        predictions = df[
            ["tweet_id", "tweet_text", "class_label"]
        ].copy()

        predictions["score_Yes"] = scores
        predictions["seuil_dev"] = configuration["seuil"]
        predictions["prediction"] = np.where(
            scores >= configuration["seuil"],
            "Yes",
            "No"
        )

        predictions.to_csv(
            OUT_F / f"F3_predictions_masque_{split}_seed{seed}.csv",
            index=False,
            encoding="utf-8-sig"
        )

        resultats_F3.extend(
            scores_groupes_F3(
                predictions, "nombres_masques", split, seed
            )
        )

    del modele
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %% CELL 164
# Partie : F3
# F3.7 — Comparer et sauvegarder les résultats
table_F3 = pd.DataFrame(resultats_F3)

resume_F3 = table_F3.groupby(
    ["variante", "split", "groupe"],
    as_index=False
).agg(
    graines=("seed", "nunique"),
    effectif=("effectif", "first"),
    Yes=("Yes", "first"),
    No=("No", "first"),
    f1_moyen=("f1_yes", "mean"),
    f1_ecart_type=("f1_yes", "std"),
    precision_moyenne=("precision_yes", "mean"),
    rappel_moyen=("rappel_yes", "mean"),
    pr_auc_moyen=("pr_auc", "mean"),
    macro_f1_moyen=("macro_f1", "mean")
)

display(resume_F3.round(4))

table_F3.to_csv(
    OUT_F / "F3_resultats_par_graine.csv",
    index=False,
    encoding="utf-8-sig"
)

resume_F3.to_csv(
    OUT_F / "F3_resume.csv",
    index=False,
    encoding="utf-8-sig"
)

global_F3 = resume_F3[resume_F3["groupe"].eq("tous")]

fig, ax = plt.subplots(figsize=(8, 4))
x = np.arange(2)
largeur = 0.35

for i, variante in enumerate(["original", "nombres_masques"]):
    donnees = (
        global_F3[global_F3["variante"].eq(variante)]
        .set_index("split")
        .loc[["dev-test", "test"]]
    )

    ax.bar(
        x + (i - 0.5) * largeur,
        donnees["f1_moyen"],
        width=largeur,
        yerr=donnees["f1_ecart_type"],
        capsize=5,
        label=variante
    )

ax.set_xticks(x, ["dev-test", "test"])
ax.set_ylim(0, 1)
ax.set_ylabel("F1 de Yes")
ax.set_title("F3 — Effet du réentraînement avec nombres masqués")
ax.legend()
ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(OUT_F / "F3_comparaison.png", dpi=300)
plt.show()

# %% CELL 171
# Partie : F4
# F4.3 — Sélectionner les Yes bien détectés
positifs_F4 = {}

for split in ["dev-test", "test"]:
    for seed in SEEDS_E:
        predictions = pd.read_csv(
            OUT_F / f"F1_predictions_{split}_seed{seed}.csv",
            dtype={"tweet_id": str}
        )

        masque = (
            predictions["class_label"].eq("Yes")
            & predictions["prediction"].eq("Yes")
        )

        positifs_F4[(split, seed)] = (
            predictions.loc[masque].reset_index(drop=True)
        )

textes_F4 = list(dict.fromkeys(
    texte
    for df in positifs_F4.values()
    for texte in df["tweet_text"]
))

entites_detectees_F4 = ner_F4(
    textes_F4,
    batch_size=16
) if textes_F4 else []

entites_F4 = dict(zip(textes_F4, entites_detectees_F4))

print("Textes distincts analysés :", len(textes_F4))

for (split, seed), df in positifs_F4.items():
    print(f"{split} — graine {seed} : {len(df)} Yes bien détectés")

# %% CELL 173
# Partie : F4
# F4.4 — Prédire après suppression et mesurer la chute
resultats_F4 = []
details_F4 = []

for seed in SEEDS_E:
    dossier = OUT_E / f"{MODELE_F1}_seed{seed}"
    configuration = CONFIGS_F1[seed]
    seuil = float(configuration["seuil"])

    tok = AutoTokenizer.from_pretrained(dossier / "tokenizer")

    modele = ModeleE(
        configuration["model_id"],
        deux_tetes=configuration["deux_tetes"]
    ).to(DEVICE_E)

    modele.load_state_dict(
        torch.load(
            dossier / "model.pt",
            map_location=DEVICE_E,
            weights_only=True
        )
    )

    preprocesseur = None

    if configuration["preprocessing"] == "ArabertPreprocessor":
        preprocesseur = ArabertPreprocessor(
            model_name=configuration["model_id"]
        )

    for split in ["dev-test", "test"]:
        df = positifs_F4[(split, seed)]

        if df.empty:
            continue

        textes_originaux = df["tweet_text"].tolist()

        variantes = [
            variantes_F4(texte, entites_F4[texte])
            for texte in textes_originaux
        ]

        for variante in [
            "sans_nombres",
            "sans_entites",
            "sans_nombres_et_entites"
        ]:
            textes_modifies = [
                exemple[variante] for exemple in variantes
            ]

            textes_entree = textes_modifies.copy()

            if preprocesseur is not None:
                textes_entree = [
                    preprocesseur.preprocess(t)
                    for t in textes_entree
                ]

            # Un texte entièrement supprimé reçoit le token inconnu.
            textes_entree = [
                t.strip() or tok.unk_token
                for t in textes_entree
            ]

            tokens = tok(
                textes_entree,
                padding="max_length",
                truncation=True,
                max_length=configuration["max_length"],
                return_tensors="pt"
            )

            scores = predire_E(modele, tokens)
            reste_yes = scores >= seuil
            scores_avant = df["score_Yes"].to_numpy()

            modifies = np.array([
                avant != apres
                for avant, apres in zip(
                    textes_originaux, textes_modifies
                )
            ])

            resultats_F4.append({
                "partie": "F4",
                "modele": MODELE_F1,
                "split": split,
                "seed": seed,
                "variante": variante,
                "seuil_dev": seuil,
                "Yes_corrects_avant": len(df),
                "tweets_modifies": int(modifies.sum()),
                "Yes_apres": int(reste_yes.sum()),
                "basculements_vers_No": int((~reste_yes).sum()),
                "chute_points": 100 * (~reste_yes).mean(),
                "score_moyen_avant": scores_avant.mean(),
                "score_moyen_apres": scores.mean(),
                "baisse_score_moyen": (scores_avant - scores).mean()
            })

            detail = df[
                ["tweet_id", "tweet_text", "class_label", "score_Yes"]
            ].copy()

            detail = detail.rename(
                columns={"score_Yes": "score_avant"}
            )

            detail["split"] = split
            detail["seed"] = seed
            detail["variante"] = variante
            detail["texte_modifie"] = textes_modifies
            detail["score_apres"] = scores
            detail["prediction_apres"] = np.where(
                reste_yes, "Yes", "No"
            )

            details_F4.append(detail)

    del modele
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# %% CELL 175
# Partie : F4
# F4.5 — Afficher et sauvegarder les résultats
table_F4 = pd.DataFrame(resultats_F4)

resume_F4 = table_F4.groupby(
    ["split", "variante"],
    as_index=False
).agg(
    graines=("seed", "nunique"),
    chute_moyenne_points=("chute_points", "mean"),
    chute_ecart_type=("chute_points", "std"),
    baisse_score_moyenne=("baisse_score_moyen", "mean")
)

display(resume_F4.round(4))

table_F4.to_csv(
    OUT_F / "F4_resultats_par_graine.csv",
    index=False,
    encoding="utf-8-sig"
)

resume_F4.to_csv(
    OUT_F / "F4_resume.csv",
    index=False,
    encoding="utf-8-sig"
)

if details_F4:
    pd.concat(details_F4, ignore_index=True).to_csv(
        OUT_F / "F4_details_tweets.csv",
        index=False,
        encoding="utf-8-sig"
    )

# %% CELL 177
# Partie : F4
# F4.6 — Tracer la chute pour la présentation
ordre = [
    "sans_nombres",
    "sans_entites",
    "sans_nombres_et_entites"
]

etiquettes = [
    "Sans nombres",
    "Sans entités",
    "Sans nombres\net entités"
]

fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

for ax, split in zip(axes, ["dev-test", "test"]):
    donnees = (
        resume_F4[resume_F4["split"].eq(split)]
        .set_index("variante")
        .reindex(ordre)
    )

    ax.bar(
        etiquettes,
        donnees["chute_moyenne_points"],
        yerr=donnees["chute_ecart_type"],
        capsize=5,
        color=["steelblue", "orange", "seagreen"]
    )

    ax.set_title(split)
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

axes[0].set_ylabel("Yes devenus No (%)")
fig.suptitle("F4 — Effet de la suppression des indices")
fig.tight_layout()

fig.savefig(OUT_F / "F4_chute.png", dpi=300)
plt.show()

# %% CELL 180
# Partie : F5
# F5.1 — Sélectionner les deux modèles sur dev
from scipy.stats import binomtest

meilleurs_F5 = (
    resume_E[
        resume_E["graines"].eq(3)
    ]
    .sort_values("f1_moyen", ascending=False)
    .head(2)
    .copy()
)

assert len(meilleurs_F5) == 2, (
    "Il faut deux configurations entraînées avec trois graines."
)

MODELES_F5 = [
    nom.removesuffix("_optimise")
    for nom in meilleurs_F5["modele"]
]

display(
    meilleurs_F5[
        ["modele", "graines", "f1_moyen", "f1_ecart_type"]
    ].round(4)
)

meilleurs_F5.to_csv(
    OUT_F / "F5_selection_sur_dev.csv",
    index=False,
    encoding="utf-8-sig"
)

# %% CELL 182
# Partie : F5
# F5.2 — Obtenir leurs prédictions
predictions_F5 = {}

for nom in MODELES_F5:
    for seed in SEEDS_E:
        dossier = OUT_E / f"{nom}_seed{seed}"

        configuration = json.loads(
            (dossier / "configuration.json").read_text()
        )

        tok = AutoTokenizer.from_pretrained(
            dossier / "tokenizer"
        )

        modele = ModeleE(
            configuration["model_id"],
            deux_tetes=configuration["deux_tetes"]
        ).to(DEVICE_E)

        modele.load_state_dict(
            torch.load(
                dossier / "model.pt",
                map_location=DEVICE_E,
                weights_only=True
            )
        )

        preprocesseur = None

        if configuration["preprocessing"] == "ArabertPreprocessor":
            preprocesseur = ArabertPreprocessor(
                model_name=configuration["model_id"]
            )

        for split, df in [("dev-test", dev_test), ("test", test)]:
            textes = df["tweet_text"].astype(str).tolist()

            if preprocesseur is not None:
                textes = [
                    preprocesseur.preprocess(t)
                    for t in textes
                ]

            textes = [
                t.strip() or tok.unk_token
                for t in textes
            ]

            tokens = tok(
                textes,
                padding="max_length",
                truncation=True,
                max_length=configuration["max_length"],
                return_tensors="pt"
            )

            scores = predire_E(modele, tokens)

            resultat = df[
                ["tweet_id", "class_label"]
            ].reset_index(drop=True).copy()

            resultat["score_Yes"] = scores
            resultat["prediction"] = (
                scores >= configuration["seuil"]
            ).astype(int)

            predictions_F5[(nom, seed, split)] = resultat

            resultat.to_csv(
                OUT_F / f"F5_predictions_{nom}_{split}_seed{seed}.csv",
                index=False,
                encoding="utf-8-sig"
            )

        del modele
        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# %% CELL 184
# Partie : F5
# F5.3 — Calculer les IC à 95 % par bootstrap
def bootstrap_f1_F5(y, predictions, repetitions=1000, seed=123):
    y = np.asarray(y, dtype=int)
    predictions = np.asarray(predictions, dtype=int)

    rng = np.random.default_rng(seed)
    scores_bootstrap = np.zeros(repetitions)

    for i in range(repetitions):
        indices = rng.integers(0, len(y), size=len(y))

        vrai = y[indices]
        predit = predictions[indices]

        tp = np.sum((vrai == 1) & (predit == 1))
        fp = np.sum((vrai == 0) & (predit == 1))
        fn = np.sum((vrai == 1) & (predit == 0))

        denominateur = 2 * tp + fp + fn

        scores_bootstrap[i] = (
            2 * tp / denominateur
            if denominateur else 0.0
        )

    borne_basse, borne_haute = np.percentile(
        scores_bootstrap, [2.5, 97.5]
    )

    return float(borne_basse), float(borne_haute)


lignes_bootstrap_F5 = []

for (nom, seed, split), df in predictions_F5.items():
    y = df["class_label"].eq("Yes").astype(int).to_numpy()
    pred = df["prediction"].to_numpy()

    borne_basse, borne_haute = bootstrap_f1_F5(
        y, pred,
        repetitions=1000,
        seed=123
    )

    lignes_bootstrap_F5.append({
        "partie": "F5",
        "modele": nom,
        "split": split,
        "seed": seed,
        "f1_yes": f1_score(y, pred, zero_division=0),
        "IC95_bas": borne_basse,
        "IC95_haut": borne_haute,
        "repetitions_bootstrap": 1000
    })

bootstrap_F5 = pd.DataFrame(lignes_bootstrap_F5)

display(bootstrap_F5.round(4))

bootstrap_F5.to_csv(
    OUT_F / "F5_bootstrap_IC95.csv",
    index=False,
    encoding="utf-8-sig"
)

# %% CELL 186
# Partie : F5
# F5.4 — Résumer les trois graines
resume_F5 = bootstrap_F5.groupby(
    ["modele", "split"],
    as_index=False
).agg(
    graines=("seed", "nunique"),
    f1_moyen=("f1_yes", "mean"),
    f1_ecart_type=("f1_yes", "std")
)

display(resume_F5.round(4))

resume_F5.to_csv(
    OUT_F / "F5_resume_3_graines.csv",
    index=False,
    encoding="utf-8-sig"
)

# %% CELL 188
# Partie : F5
# F5.5 — Effectuer le test de McNemar
modele_A, modele_B = MODELES_F5

lignes_mcnemar_F5 = []

for split in ["dev-test", "test"]:
    for seed in SEEDS_E:
        a = predictions_F5[(modele_A, seed, split)]
        b = predictions_F5[(modele_B, seed, split)]

        assert np.array_equal(
            a["tweet_id"].to_numpy(),
            b["tweet_id"].to_numpy()
        ), "Les tweets doivent être dans le même ordre."

        assert np.array_equal(
            a["class_label"].to_numpy(),
            b["class_label"].to_numpy()
        )

        y = a["class_label"].eq("Yes").astype(int).to_numpy()

        correct_A = a["prediction"].to_numpy() == y
        correct_B = b["prediction"].to_numpy() == y

        a_seul = int(np.sum(correct_A & ~correct_B))
        b_seul = int(np.sum(~correct_A & correct_B))

        discordances = a_seul + b_seul

        p_value = (
            binomtest(
                a_seul,
                n=discordances,
                p=0.5,
                alternative="two-sided"
            ).pvalue
            if discordances else 1.0
        )

        lignes_mcnemar_F5.append({
            "split": split,
            "seed": seed,
            "modele_A": modele_A,
            "modele_B": modele_B,
            "deux_corrects": int(np.sum(correct_A & correct_B)),
            "A_seul_correct": a_seul,
            "B_seul_correct": b_seul,
            "deux_incorrects": int(np.sum(~correct_A & ~correct_B)),
            "p_value": p_value
        })

mcnemar_F5 = pd.DataFrame(lignes_mcnemar_F5)

# %% CELL 190
# Partie : F5
# F5.6 — Corriger les comparaisons multiples et sauvegarder
p_values = mcnemar_F5["p_value"].to_numpy()
ordre = np.argsort(p_values)
nombre_tests = len(p_values)

p_corrigees_triees = np.minimum(
    1.0,
    np.maximum.accumulate(
        (nombre_tests - np.arange(nombre_tests))
        * p_values[ordre]
    )
)

p_corrigees = np.empty(nombre_tests)
p_corrigees[ordre] = p_corrigees_triees

mcnemar_F5["p_value_Holm"] = p_corrigees
mcnemar_F5["significatif_5pct"] = p_corrigees < 0.05

display(mcnemar_F5.round(4))

mcnemar_F5.to_csv(
    OUT_F / "F5_mcnemar.csv",
    index=False,
    encoding="utf-8-sig"
)

# %% CELL 193
# Partie : F6
# F6.1 — Sélectionner 30 erreurs
SEED_F6 = 42
FICHIER_F6 = OUT_F / f"F6_30_erreurs_{MODELE_F1}.csv"

CATEGORIES_F6 = {
    "1": "opinion déguisée en fait",
    "2": "affirmation banale",
    "3": "affirmation invérifiable",
    "4": "contexte manquant",
    "5": "bruit d'annotation"
}

# Ne pas écraser les annotations déjà enregistrées.
if FICHIER_F6.exists():
    erreurs_F6 = pd.read_csv(
        FICHIER_F6,
        dtype={"tweet_id": str},
        keep_default_na=False
    )

else:
    predictions = pd.read_csv(
        OUT_F / f"F1_predictions_test_seed{SEED_F6}.csv",
        dtype={"tweet_id": str}
    )

    erreurs = predictions[
        predictions["class_label"] != predictions["prediction"]
    ].copy()

    erreurs["type_erreur"] = np.where(
        erreurs["class_label"].eq("No"),
        "FP",
        "FN"
    )

    assert len(erreurs) >= 30, "Moins de 30 erreurs disponibles."

    morceaux = []

    for type_erreur in ["FP", "FN"]:
        groupe = erreurs[erreurs["type_erreur"].eq(type_erreur)]

        morceaux.append(
            groupe.sample(
                n=min(15, len(groupe)),
                random_state=SEED_F6
            )
        )

    selection = pd.concat(morceaux)

    # Compléter si un type contient moins de 15 erreurs.
    manque = 30 - len(selection)

    if manque > 0:
        restantes = erreurs.drop(index=selection.index)

        selection = pd.concat([
            selection,
            restantes.sample(n=manque, random_state=SEED_F6)
        ])

    erreurs_F6 = selection.reset_index(drop=True)
    erreurs_F6["categorie"] = ""
    erreurs_F6["justification"] = ""

    erreurs_F6.to_csv(
        FICHIER_F6,
        index=False,
        encoding="utf-8-sig"
    )

print("Nombre d’erreurs :", len(erreurs_F6))
display(erreurs_F6["type_erreur"].value_counts())

# %% CELL 195
# Partie : F6
# F6.2 — Lire et classer les erreurs
erreurs_F6 = pd.read_csv(
    FICHIER_F6,
    dtype={"tweet_id": str},
    keep_default_na=False
)

print(
    "Erreurs déjà annotées :",
    erreurs_F6["categorie"].str.strip().ne("").sum()
)

# %% CELL 196
# Partie : F6
# F6.2 — Lire et classer les erreurs
for i in erreurs_F6.index:
    # Reprendre uniquement les erreurs non encore annotées.
    if str(erreurs_F6.at[i, "categorie"]).strip():
        continue

    ligne = erreurs_F6.loc[i]

    print("\n" + "=" * 60)
    print(f"Erreur {i + 1}/{len(erreurs_F6)}")
    print("Tweet :", ligne["tweet_text"])
    print("Label officiel :", ligne["class_label"])
    print("Prédiction :", ligne["prediction"])
    print("Type :", ligne["type_erreur"])

    print("\nCatégories :")
    for numero, categorie in CATEGORIES_F6.items():
        print(f"{numero} — {categorie}")

    choix = input("\nNuméro de catégorie, ou q pour arrêter : ").strip()

    while choix not in CATEGORIES_F6 and choix.lower() != "q":
        choix = input("Saisis 1, 2, 3, 4, 5 ou q : ").strip()

    if choix.lower() == "q":
        break

    justification = input(
        "Explique brièvement ton choix : "
    ).strip()

    while not justification:
        justification = input(
            "Ajoute une justification : "
        ).strip()

    erreurs_F6.at[i, "categorie"] = CATEGORIES_F6[choix]
    erreurs_F6.at[i, "justification"] = justification

    # Sauvegarde après chaque réponse.
    erreurs_F6.to_csv(
        FICHIER_F6,
        index=False,
        encoding="utf-8-sig"
    )

print("Annotations sauvegardées dans :", FICHIER_F6)

# %% CELL 198
# Partie : F6
# F6.3 — Résumer les catégories
erreurs_F6 = pd.read_csv(
    FICHIER_F6,
    dtype={"tweet_id": str},
    keep_default_na=False
)

annotees_F6 = erreurs_F6[
    erreurs_F6["categorie"].str.strip().ne("")
].copy()

print(f"Erreurs classées : {len(annotees_F6)}/30")

comptages = (
    annotees_F6["categorie"]
    .value_counts()
    .reindex(list(CATEGORIES_F6.values()), fill_value=0)
)

resume_F6 = comptages.rename_axis("categorie").reset_index(
    name="nombre"
)

resume_F6["pourcentage"] = (
    100 * resume_F6["nombre"] / len(annotees_F6)
    if len(annotees_F6)
    else np.nan
)

display(resume_F6.round(2))

if not annotees_F6.empty:
    display(pd.crosstab(
        annotees_F6["categorie"],
        annotees_F6["type_erreur"]
    ))

resume_F6.to_csv(
    OUT_F / "F6_resume_categories.csv",
    index=False,
    encoding="utf-8-sig"
)

# %% CELL 200
# Partie : F6
# F6.4 — Tracer les catégories pour la présentation
if not annotees_F6.empty:
    fig, ax = plt.subplots(figsize=(9, 4))

    ax.barh(
        resume_F6["categorie"],
        resume_F6["nombre"],
        color="steelblue"
    )

    ax.invert_yaxis()
    ax.set_xlabel("Nombre d’erreurs")
    ax.set_title(
        f"F6 — Catégories des erreurs annotées "
        f"({len(annotees_F6)}/30)"
    )

    for position, nombre in enumerate(resume_F6["nombre"]):
        ax.text(nombre + 0.1, position, str(nombre), va="center")

    ax.set_xlim(0, max(resume_F6["nombre"]) + 2)
    fig.tight_layout()

    fig.savefig(
        OUT_F / "F6_categories_erreurs.png",
        dpi=300
    )

    plt.show()

# %% CELL 202
# Partie : G
# PARTIE G
# Correctif B1 : prédictions par lots sur GPU.
# Les appels avec un seul tweet restent sur CPU pour la latence G.

import torch
import numpy as np
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

assert "charger_modele_G" in globals(), (
    "Exécute d'abord la cellule qui définit charger_modele_G(), "
    "sans lancer sa boucle d'évaluation."
)

if not torch.cuda.is_available():
    raise RuntimeError(
        "Aucun GPU disponible. Le correctif ne peut pas accélérer B1 "
        "sur GPU. Sauvegarde tes résultats avant de changer "
        "le type d'environnement Colab."
    )

# Conserver la fonction originale, même si cette cellule est réexécutée.
if "_charger_G_original" not in globals():
    _charger_G_original = charger_modele_G


def charger_modele_G(ligne):
    # Les autres parties gardent leur chargement habituel.
    if ligne["partie"] != "B1":
        return _charger_G_original(ligne)

    langue = ligne["modele"].split("_")[1]
    hypotheses = HYPOTHESES[langue]

    identifiant = (
        "MoritzLaurer/"
        "mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
    )

    print(f"Chargement B1 ({langue}) sur GPU...", flush=True)

    tok = AutoTokenizer.from_pretrained(identifiant)

    modele = AutoModelForSequenceClassification.from_pretrained(
        identifiant
    ).to("cuda").eval()

    indices = [
        int(i)
        for i, label in modele.config.id2label.items()
        if "entail" in label.lower()
    ]

    assert len(indices) == 1, modele.config.id2label
    entailment = indices[0]

    taille = sum(
        t.numel() * t.element_size()
        for t in modele.state_dict().values()
    ) / 1024**2

    appareil_actuel = torch.device("cuda")

    @torch.inference_mode()
    def predire(textes):
        nonlocal appareil_actuel

        # Dans G :
        # - plusieurs tweets = évaluation accélérée sur GPU ;
        # - un tweet = benchmark CPU demandé.
        appareil = torch.device(
            "cpu" if len(textes) == 1 else "cuda"
        )

        if appareil != appareil_actuel:
            modele.to(appareil)
            appareil_actuel = appareil

            if appareil.type == "cpu":
                torch.cuda.empty_cache()
                print("B1 : passage au benchmark CPU.", flush=True)

        scores = []

        # Quatre tweets = huit paires prémisse/hypothèse.
        for debut in range(0, len(textes), 4):
            premises = []
            hypotheses_batch = []

            for texte in textes[debut:debut + 4]:
                premises.extend([str(texte), str(texte)])
                hypotheses_batch.extend([
                    hypotheses["Yes"],
                    hypotheses["No"]
                ])

            inputs = tok(
                premises,
                hypotheses_batch,
                padding=True,
                truncation="only_first",
                max_length=512,
                return_tensors="pt"
            ).to(appareil)

            logits = modele(**inputs).logits
            logits = logits[:, entailment].reshape(-1, 2)

            scores.extend(
                logits.softmax(dim=1)[:, 0].cpu().tolist()
            )

        return np.asarray(scores)

    print("B1 prêt : GPU pour les scores, CPU pour la latence.", flush=True)
    return predire, taille


# Limiter G au meilleur modèle de chaque partie.
classement = (
    experiences_G
    .groupby(["partie", "modele"], as_index=False)
    .agg(f1_dev_moyen=("f1_yes", "mean"))
)

selection = (
    classement
    .sort_values("f1_dev_moyen", ascending=False)
    .drop_duplicates("partie")
)

experiences_G = (
    experiences_G
    .merge(
        selection[["partie", "modele"]],
        on=["partie", "modele"],
        how="inner"
    )
    .sort_values(["partie", "modele", "seed"])
)

display(selection.sort_values("partie").round(4))
print("Correctif appliqué. Relance uniquement la cellule G.")

# %% CELL 203
# Partie : G
# PARTIE G
# ============================================================
# G OPTIMISÉ — remplacer les anciennes sections 3, 4 et 5
# Prérequis : experiences_G et charger_modele_G déjà définis.
# Aucun entraînement.
# ============================================================

import hashlib

OUT_G.mkdir(parents=True, exist_ok=True)

# Augmenter cette version si tu modifies le prétraitement,
# l'architecture ou la méthode de mesure de latence.
VERSION_G = 1

FICHIER_DETAILS_G = OUT_G / "G_resultats_par_graine.csv"
FICHIER_CACHE_G = OUT_G / "G_cache.json"

cache_G = (
    json.loads(FICHIER_CACHE_G.read_text())
    if FICHIER_CACHE_G.exists()
    else {}
)

details_existants_G = (
    pd.read_csv(FICHIER_DETAILS_G)
    if FICHIER_DETAILS_G.exists()
    else pd.DataFrame()
)

# Même échantillon de 30 tweets pour toutes les latences.
textes_latence_G = (
    test["tweet_text"]
    .sample(n=min(30, len(test)), random_state=42)
    .astype(str)
    .tolist()
)

# Identifier les données utilisées pour éviter un mauvais cache.
empreinte_donnees_G = hashlib.sha256(
    pd.concat([
        dev_test.assign(split="dev-test"),
        test.assign(split="test")
    ])[["split", "tweet_id", "tweet_text", "class_label"]]
    .to_csv(index=False).encode("utf-8")
).hexdigest()

empreinte_A_G = hashlib.sha256(
    pickle.dumps(baseline_A)
).hexdigest()


def signature_G(ligne):
    """Identifier la configuration et les checkpoints actuels."""
    partie = ligne["partie"]
    base = ligne["modele"].removesuffix("_optimise")
    seed = int(ligne["seed"])

    fichiers = []

    if partie == "C":
        fichiers = [
            OUT_C / f"{base}_{seed}.pt",
            OUT_C / "vocabulaire.json"
        ]

    elif partie == "D":
        fichiers = [
            OUT_D / (
                f"{base}.joblib"
                if base.startswith("D1")
                else "D2.pt"
            )
        ]

    elif partie in ["E1", "E2"]:
        dossier = OUT_E / f"{base}_seed{seed}"
        fichiers = [
            dossier / "model.pt",
            dossier / "configuration.json"
        ]

    contenu = {
        "version": VERSION_G,
        "donnees": empreinte_donnees_G,
        "partie": partie,
        "modele": ligne["modele"],
        "seed": seed,
        "seuil": (
            float(ligne["seuil"])
            if pd.notna(ligne["seuil"]) else 0.5
        ),
        "f1_dev": float(ligne["f1_yes"]),
        "fichiers": [
            (str(p), p.stat().st_size, p.stat().st_mtime_ns)
            for p in fichiers
        ],
        "baseline_A": empreinte_A_G if partie == "A" else None,
        "hypotheses": HYPOTHESES if partie == "B1" else None
    }

    return hashlib.sha256(
        json.dumps(
            contenu, sort_keys=True, ensure_ascii=False
        ).encode("utf-8")
    ).hexdigest()


def lire_scores_G(fichier, df):
    """Accepter un CSV uniquement si les tweets correspondent."""
    if not fichier.exists():
        return None

    sauvegarde = pd.read_csv(
        fichier, dtype={"tweet_id": str}
    )

    if (
        len(sauvegarde) != len(df)
        or "score_Yes" not in sauvegarde
        or not np.array_equal(
            sauvegarde["tweet_id"].astype(str).to_numpy(),
            df["tweet_id"].astype(str).to_numpy()
        )
        or not np.array_equal(
            sauvegarde["class_label"].to_numpy(),
            df["class_label"].to_numpy()
        )
    ):
        return None

    if "tweet_text" in sauvegarde:
        if not np.array_equal(
            sauvegarde["tweet_text"].astype(str).to_numpy(),
            df["tweet_text"].astype(str).to_numpy()
        ):
            return None

    scores = sauvegarde["score_Yes"].to_numpy(dtype=float)

    return scores if np.isfinite(scores).all() else None


resultats_G = []
confusion_G = None

nom_retenu_G = (
    MODELE_F1
    if MODELE_F1.endswith("_optimise")
    else MODELE_F1 + "_optimise"
)

# Trier pour traiter la graine 42 avant les autres.
experiences_G = experiences_G.sort_values(
    ["partie", "modele", "seed"]
)

for _, ligne in experiences_G.iterrows():
    partie = ligne["partie"]
    nom = ligne["modele"]
    base = nom.removesuffix("_optimise")
    seed = int(ligne["seed"])
    cle = f"{partie}|{nom}|{seed}"

    signature = signature_G(ligne)
    cache_valide = cache_G.get(cle, {}).get("signature") == signature

    seuil = (
        float(ligne["seuil"])
        if pd.notna(ligne["seuil"]) else 0.5
    )

    print(f"\n{partie} | {nom} | graine {seed}")

    # Réutiliser une expérience G entièrement terminée.
    precedente = pd.DataFrame()

    if cache_valide and not details_existants_G.empty:
        precedente = details_existants_G[
            details_existants_G["partie"].eq(partie)
            & details_existants_G["modele"].eq(nom)
            & details_existants_G["seed"].eq(seed)
        ]

    if not precedente.empty:
        print("Résultat G déjà calculé : réutilisé.")
        resultat = precedente.iloc[-1].to_dict()

        # Le temps du journal peut avoir été complété depuis.
        resultat["temps_entrainement_s"] = (
            0.0 if partie in ["B1", "B2"]
            else ligne.get("temps_entrainement_s", np.nan)
        )

        resultats_G.append(resultat)

        if nom == nom_retenu_G and seed == 42:
            df = test
            scores = lire_scores_G(
                OUT_G / f"G_predictions_{nom}_test_seed{seed}.csv",
                df
            )

            if scores is not None:
                confusion_G = confusion_matrix(
                    df["class_label"].eq("Yes").astype(int),
                    scores >= seuil,
                    labels=[0, 1]
                )

        continue

    # Charger le modèle seulement si une prédiction ou
    # une mesure CPU est réellement nécessaire.
    predire = None
    taille = np.nan
    mesures = {}

    for split, df in [("dev-test", dev_test), ("test", test)]:
        fichier_G = (
            OUT_G / f"G_predictions_{nom}_{split}_seed{seed}.csv"
        )

        scores = (
            lire_scores_G(fichier_G, df)
            if cache_valide else None
        )

        # Réutiliser F5 ou F1 pour les mêmes checkpoints.
        if scores is None and partie in ["E1", "E2"]:
            checkpoint = OUT_E / f"{base}_seed{seed}" / "model.pt"

            candidats = [
                OUT_F / f"F5_predictions_{base}_{split}_seed{seed}.csv"
            ]

            if base == MODELE_F1:
                candidats.append(
                    OUT_F / f"F1_predictions_{split}_seed{seed}.csv"
                )

            for fichier in candidats:
                # Ignorer les prédictions antérieures au checkpoint.
                if (
                    fichier.exists()
                    and fichier.stat().st_mtime_ns
                    >= checkpoint.stat().st_mtime_ns
                ):
                    scores = lire_scores_G(fichier, df)

                    if scores is not None:
                        print(f"{split} : prédictions sauvegardées.")
                        break

        if scores is None:
            if predire is None:
                predire, taille = charger_modele_G(ligne)

            textes = df["tweet_text"].astype(str).tolist()

            scores = np.concatenate([
                predire(textes[i:i + 16])
                for i in range(0, len(textes), 16)
            ])

            print(f"{split} : nouvelles prédictions, sans entraînement.")

        y = df["class_label"].eq("Yes").astype(int).to_numpy()
        pred = (scores >= seuil).astype(int)

        mesures[split] = threshold_and_metrics(
            y, scores, threshold=seuil
        )

        sortie = df[
            ["tweet_id", "tweet_text", "class_label"]
        ].reset_index(drop=True).copy()

        sortie["score_Yes"] = scores
        sortie["prediction"] = np.where(pred == 1, "Yes", "No")

        sortie.to_csv(
            fichier_G, index=False, encoding="utf-8-sig"
        )

        if nom == nom_retenu_G and seed == 42 and split == "test":
            confusion_G = confusion_matrix(y, pred, labels=[0, 1])

    # Mesurer la latence une fois par configuration, graine 42.
    latence = np.nan

    if seed == 42:
        if predire is None:
            predire, taille = charger_modele_G(ligne)

        for texte in textes_latence_G[:3]:
            predire([texte])  # chauffe

        durees = []

        for texte in textes_latence_G:
            debut = perf_counter()
            _ = predire([texte]) >= seuil
            durees.append(1000 * (perf_counter() - debut))

        latence = float(np.mean(durees))

    # Même architecture : taille reprise de la graine 42.
    elif pd.isna(taille):
        reference = next(
            (
                r for r in resultats_G
                if r["modele"] == nom and int(r["seed"]) == 42
            ),
            None
        )

        if reference is not None:
            taille = reference["taille_poids_Mio"]
        else:
            if predire is None:
                predire, taille = charger_modele_G(ligne)

    temps = (
        0.0 if partie in ["B1", "B2"]
        else pd.to_numeric(
            ligne.get("temps_entrainement_s", np.nan),
            errors="coerce"
        )
    )

    resultat = {
        "partie": partie,
        "modele": nom,
        "seed": seed,
        "seuil": seuil,
        "f1_dev": float(ligne["f1_yes"]),
        "f1_dev_test": mesures["dev-test"]["f1_yes"],
        "f1_test": mesures["test"]["f1_yes"],
        "pr_auc_test": mesures["test"]["pr_auc"],
        "macro_f1_test": mesures["test"]["macro_f1"],
        "ecart_F1_test_moins_dev_test": (
            mesures["test"]["f1_yes"]
            - mesures["dev-test"]["f1_yes"]
        ),
        "temps_entrainement_s": temps,
        "latence_CPU_ms_exemple": latence,
        "taille_poids_Mio": taille
    }

    resultats_G.append(resultat)

    # Conserver aussi les expériences précédentes non encore parcourues.
    nouveaux = pd.DataFrame(resultats_G)

    details_existants_G = pd.concat(
        [details_existants_G, nouveaux], ignore_index=True
    ).drop_duplicates(
        ["partie", "modele", "seed"], keep="last"
    )

    details_existants_G.to_csv(
        FICHIER_DETAILS_G,
        index=False,
        encoding="utf-8-sig"
    )

    cache_G[cle] = {"signature": signature}

    FICHIER_CACHE_G.write_text(
        json.dumps(cache_G, indent=2),
        encoding="utf-8"
    )

    if predire is not None:
        del predire

    gc.collect()


# ============================================================
# Tableau final : toutes les configurations
# ============================================================

details_G = pd.DataFrame(resultats_G)

tableau_G = details_G.groupby(
    ["partie", "modele"], as_index=False
).agg(
    graines=("seed", "nunique"),
    f1_dev_moyen=("f1_dev", "mean"),
    f1_dev_test_moyen=("f1_dev_test", "mean"),
    f1_test_moyen=("f1_test", "mean"),
    f1_test_ecart_type=("f1_test", "std"),
    pr_auc_test_moyen=("pr_auc_test", "mean"),
    macro_f1_test_moyen=("macro_f1_test", "mean"),
    ecart_test_moins_dev_test=(
        "ecart_F1_test_moins_dev_test", "mean"
    ),
    temps_entrainement_moyen_s=("temps_entrainement_s", "mean"),
    latence_CPU_ms_exemple=("latence_CPU_ms_exemple", "mean"),
    taille_poids_Mio=("taille_poids_Mio", "first")
)

display(tableau_G.round(4))

details_G.to_csv(
    FICHIER_DETAILS_G, index=False, encoding="utf-8-sig"
)

tableau_G.to_csv(
    OUT_G / "G_tableau_final_tous_modeles.csv",
    index=False,
    encoding="utf-8-sig"
)

# Mettre à jour le journal.
entrees = details_G.copy()
entrees["experience_source"] = entrees["partie"]
entrees["partie"] = "G"
entrees["split"] = "test"

entrees = entrees.rename(columns={
    "f1_test": "f1_yes",
    "pr_auc_test": "pr_auc",
    "macro_f1_test": "macro_f1"
})

journal = pd.read_csv(RESULTS / "results.csv")

pd.concat(
    [journal, entrees], ignore_index=True
).drop_duplicates(
    ["partie", "modele", "seed", "split"], keep="last"
).to_csv(
    RESULTS / "results.csv",
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# Matrice de confusion du modèle retenu, graine 42
# ============================================================

assert confusion_G is not None, (
    "Prédictions du modèle retenu absentes : vérifier son CSV G."
)

fig, ax = plt.subplots(figsize=(5, 4))

ConfusionMatrixDisplay(
    confusion_G,
    display_labels=["No", "Yes"]
).plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title(f"Test — {MODELE_F1}\nGraine 42")
ax.set_xlabel("Label prédit")
ax.set_ylabel("Label réel")

fig.tight_layout()
fig.savefig(OUT_G / "G_matrice_confusion.png", dpi=300)
plt.show()

pd.DataFrame(
    confusion_G,
    index=["reel_No", "reel_Yes"],
    columns=["predit_No", "predit_Yes"]
).to_csv(
    OUT_G / "G_matrice_confusion.csv",
    encoding="utf-8-sig"
)

print("Résultats sauvegardés dans :", OUT_G)

