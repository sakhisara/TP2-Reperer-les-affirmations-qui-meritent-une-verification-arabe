"""Code data extrait du notebook. Exécution ordonnée via run.py."""

# %% CELL 2
# Partie : AUDIT
# 1. Télécharger et charger les données
from pathlib import Path
from zipfile import ZipFile
import subprocess
import pandas as pd
import numpy as np
import re
import unicodedata

REPO = Path("/content/clef2024-checkthat-lab")

if not REPO.exists():
    subprocess.run(
        [
            "git", "clone", "--depth", "1",
            "https://gitlab.com/checkthat_lab/clef2024-checkthat-lab.git",
            str(REPO)
        ],
        check=True
    )

# On lit seulement train et dev, sans extraire les autres splits.
archive = REPO / "task1/data/CT24_checkworthy_arabic.zip"

with ZipFile(archive) as z:
    def load_split(split):
        filename = f"CT24_checkworthy_arabic_{split}.tsv"
        with z.open(filename) as f:
            return pd.read_csv(f, sep="\t", dtype=str)

    train = load_split("train")
    val = load_split("dev")

for name, df in [("train", train), ("val", val)]:
    assert df["tweet_text"].notna().all()
    assert df["class_label"].isin(["Yes", "No"]).all()
    assert not df["tweet_text"].str.strip().eq("").any()

    print(f"{name} : {len(df)} tweets")
    print(df["class_label"].value_counts(), "\n")

display(train.head(3))

# %% CELL 6
# Partie : AUDIT
# 3. Produire le tableau des comptes
from pathlib import Path

RESULTS = Path("/content/results")
RESULTS.mkdir(parents=True, exist_ok=True)

# %% CELL 95
# Partie : E
# 4 — Charger le train anglais
archive_en = Path(
    "/content/clef2024-checkthat-lab/"
    "task1/data/CT24_checkworthy_english.zip"
)

with ZipFile(archive_en) as z:
    nom_fichier = next(
        n for n in z.namelist()
        if n.endswith("_train.tsv")
        and not n.startswith("__MACOSX")
    )

    with z.open(nom_fichier) as f:
        EN_TRAIN = pd.read_csv(f, sep="\t", dtype=str)

EN_TRAIN = EN_TRAIN.rename(columns={
    "Text": "tweet_text",
    "Sentence_id": "tweet_id"
})

assert EN_TRAIN["tweet_text"].notna().all()
assert EN_TRAIN["class_label"].isin(["No", "Yes"]).all()

nombre_en_initial = len(EN_TRAIN)

# Exclure les textes identiques avec labels contradictoires.
labels_en = EN_TRAIN.groupby(
    "tweet_text"
)["class_label"].nunique()

textes_contradictoires = labels_en[labels_en > 1].index

EN_TRAIN = EN_TRAIN[
    ~EN_TRAIN["tweet_text"].isin(textes_contradictoires)
]

# Garder une occurrence par texte exact.
EN_TRAIN = EN_TRAIN.drop_duplicates(
    "tweet_text"
).reset_index(drop=True)

Y_EN = EN_TRAIN["class_label"].map(
    {"No": 0, "Yes": 1}
).to_numpy()

print("Train anglais original :", nombre_en_initial)
print("Après traitement :", len(EN_TRAIN))

# %% CELL 134
# Partie : F1
# F1.2 — Ouvrir dev-test et le test officiel
DOSSIER_DATA = Path(
    "/content/clef2024-checkthat-lab/task1/data"
)

with ZipFile(
    DOSSIER_DATA / "CT24_checkworthy_arabic.zip"
) as z:
    with z.open(
        "CT24_checkworthy_arabic_dev-test.tsv"
    ) as f:
        dev_test = pd.read_csv(f, sep="\t", dtype=str)

with ZipFile(
    DOSSIER_DATA / "CT24_checkworthy_test_gold.zip"
) as z:
    nom_test = next(
        nom for nom in z.namelist()
        if nom.endswith("arabic_test_gold.tsv")
        and not nom.startswith("__MACOSX")
    )

    with z.open(nom_test) as f:
        test = pd.read_csv(f, sep="\t", dtype=str)

assert len(dev_test) == 500
assert len(test) == 610

for df in [dev_test, test]:
    assert df["tweet_text"].notna().all()
    assert df["class_label"].isin(["No", "Yes"]).all()

# Comparer les proportions de Yes.
distribution_F1 = pd.DataFrame([
    {
        "split": nom,
        "effectif": len(df),
        "pourcentage_Yes": 100 * df["class_label"].eq("Yes").mean()
    }
    for nom, df in [
        ("dev", AR_DEV),
        ("dev-test", dev_test),
        ("test", test)
    ]
])

display(distribution_F1.round(2))

distribution_F1.to_csv(
    OUT_F / "F1_distribution_labels.csv",
    index=False,
    encoding="utf-8-sig"
)

