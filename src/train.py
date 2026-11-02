"""Code train extrait du notebook. Exécution ordonnée via run.py."""

# %% CELL 27
# Partie : A
# 6 — Entraîner TF-IDF + régression logistique
baseline_A = Pipeline([
    (
        "tfidf",
        FeatureUnion([
            (
                "mots",
                TfidfVectorizer(
                    preprocessor=normaliser_arabe,
                    lowercase=False,
                    ngram_range=(1, 2),
                    min_df=2,
                    sublinear_tf=True,
                    dtype=np.float32
                )
            ),
            (
                "caracteres",
                TfidfVectorizer(
                    preprocessor=normaliser_arabe,
                    lowercase=False,
                    analyzer="char",
                    ngram_range=(2, 5),
                    min_df=2,
                    sublinear_tf=True,
                    dtype=np.float32
                )
            )
        ])
    ),
    (
        "lr",
        LogisticRegression(
            class_weight="balanced",
            solver="liblinear",
            max_iter=2000,
            random_state=SEED
        )
    )
])

debut = perf_counter()

# Seul train est utilisé pour apprendre TF-IDF et la régression.
baseline_A.fit(train_A["tweet_text"], y_train)

temps_A = perf_counter() - debut

# Dev est seulement transformé et évalué.
position_yes = list(baseline_A.classes_).index(1)

scores_val_A = baseline_A.predict_proba(
    val_A["tweet_text"]
)[:, position_yes]

print(f"Temps d’entraînement : {temps_A:.2f} secondes")

evaluer_A(
    "TFIDF_LR_seuil_050",
    predictions=(scores_val_A >= 0.5).astype(int),
    scores=scores_val_A,
    seuil=0.5,
    temps_entrainement=temps_A,
    notes="Mots 1-2 + caractères 2-5 ; balanced ; seuil standard."
)

# %% CELL 61
# Partie : C
# 4. Entraîner, évaluer et sauvegarder
from time import perf_counter
def mesurer_C(scores, seuil=None):
    p, r, seuils = precision_recall_curve(ydev, scores)
    if seuil is None:
        f = np.divide(2*p[:-1]*r[:-1], p[:-1]+r[:-1], out=np.zeros_like(p[:-1]), where=(p[:-1]+r[:-1])!=0)
        candidats = np.flatnonzero(np.isclose(f, f.max()))
        seuil = float(seuils[candidats[np.argmin(abs(seuils[candidats]-.5))]])
    pred = scores >= seuil
    return dict(seuil=seuil, precision_yes=precision_score(ydev,pred,zero_division=0),
        rappel_yes=recall_score(ydev,pred,zero_division=0), f1_yes=f1_score(ydev,pred,zero_division=0),
        macro_f1=f1_score(ydev,pred,average='macro',zero_division=0), pr_auc=auc(r,p))

@torch.inference_mode()
def predire_C(modele, x):
    modele.eval()
    return np.concatenate([modele(b.to(DEVICE)).sigmoid().cpu().numpy()
        for (b,) in DataLoader(TensorDataset(x), batch_size=BATCH)])

resultats_C = []

def entrainer_C(
    nom, arch, embedding="random", positions=True,
    strategie="standard", depart=None
):
    seed_everything(SEED)

    modele = (
        CharCNNC()
        if arch == "C4"
        else ModeleMotsC(
            arch,
            ALEATOIRE if embedding == "random" else ARAVEC,
            positions
        )
    )

    modele = modele.to(DEVICE)

    if depart:
        modele.load_state_dict(torch.load(
            depart,
            map_location=DEVICE,
            weights_only=True
        ))

    if arch != "C4":
        modele.emb.weight.requires_grad_(embedding != "fige")

    xt, xv = (XC, VC) if arch == "C4" else (XM, VM)

    effectifs = torch.bincount(ytr.long(), minlength=2)
    sampler = None

    if strategie == "surechantillonnage":
        sampler = WeightedRandomSampler(
            (1 / effectifs.double())[ytr.long()],
            len(ytr),
            replacement=True,
            generator=torch.Generator().manual_seed(SEED)
        )

    loader = DataLoader(
        TensorDataset(xt, ytr),
        batch_size=BATCH,
        shuffle=sampler is None,
        sampler=sampler
    )

    poids = (
        effectifs[0] / effectifs[1]
        if strategie == "ponderee"
        else torch.tensor(1.)
    )

    perte = nn.BCEWithLogitsLoss(pos_weight=poids.to(DEVICE))

    groupes = [{
        "params": [
            p for n, p in modele.named_parameters()
            if not n.startswith("emb.")
        ],
        "lr": 1e-3
    }]

    if modele.emb.weight.requires_grad:
        groupes.append({
            "params": modele.emb.parameters(),
            "lr": 1e-4 if embedding == "degele" else 1e-3
        })

    optim = torch.optim.Adam(groupes)

    meilleur, patience = -1., 0
    fichier = OUT_C / f"{nom}_{SEED}.pt"

    # ---------- DÉBUT DU CHRONOMÉTRAGE ----------
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    debut_entrainement = perf_counter()

    for epoch in range(EPOCHS):
        modele.train()

        for x, y in loader:
            optim.zero_grad()

            loss = perte(
                modele(x.to(DEVICE)),
                y.to(DEVICE)
            )

            loss.backward()
            nn.utils.clip_grad_norm_(modele.parameters(), 1.)
            optim.step()

        f1 = mesurer_C(predire_C(modele, xv))["f1_yes"]

        print(
            f"{nom} | époque {epoch + 1} "
            f"| F1-Yes dev : {f1:.4f}"
        )

        if f1 > meilleur + 1e-6:
            meilleur, patience = f1, 0
            torch.save(modele.state_dict(), fichier)
        else:
            patience += 1

        if patience >= 3:
            break

    # ---------- FIN DU CHRONOMÉTRAGE ----------
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    temps_phase = perf_counter() - debut_entrainement

    # Si ce modèle reprend un checkpoint, ajouter le temps
    # déjà nécessaire pour entraîner ce checkpoint.
    temps_precedent = 0.0

    if depart:
        nom_precedent = Path(depart).stem.removesuffix(f"_{SEED}")

        precedents = [
            r for r in resultats_C
            if r["modele"] == f"{nom_precedent}_optimise"
            and r["seed"] == SEED
        ]

        temps_precedent = (
            precedents[-1]["temps_entrainement_s"]
            if precedents else float("nan")
        )

    temps_total = temps_precedent + temps_phase

    print(f"Temps de cette phase : {temps_phase:.2f} secondes")
    print(f"Temps total du modèle : {temps_total:.2f} secondes")

    # Recharger le meilleur modèle.
    modele.load_state_dict(torch.load(
        fichier,
        map_location=DEVICE,
        weights_only=True
    ))

    scores = predire_C(modele, xv)

    # Enregistrer les deux évaluations avec leur durée.
    nouvelles_lignes = []

    for variante, seuil in [("050", .5), ("optimise", None)]:
        nouvelles_lignes.append({
            "partie": "C",
            "modele": f"{nom}_{variante}",
            "split": "dev",
            "seed": SEED,
            "temps_entrainement_s": temps_total,
            "temps_phase_s": temps_phase,
            **mesurer_C(scores, seuil)
        })

    resultats_C.extend(nouvelles_lignes)

    chemin = RESULTS / "results.csv"

    historique = (
        pd.read_csv(chemin)
        if chemin.exists()
        else pd.DataFrame()
    )

    nouveaux = pd.DataFrame(nouvelles_lignes)

    if not historique.empty:
        remplacer = (
            historique["partie"].eq("C")
            & historique["seed"].eq(SEED)
            & historique["split"].eq("dev")
            & historique["modele"].isin(nouveaux["modele"])
        )

        historique = historique.loc[~remplacer]

    pd.concat(
        [historique, nouveaux],
        ignore_index=True
    ).to_csv(
        chemin,
        index=False,
        encoding="utf-8-sig"
    )

    resultat = {
        "nom": nom,
        "arch": arch,
        "embedding": embedding,
        "positions": positions,
        "checkpoint": str(fichier),
        "f1": mesurer_C(scores)["f1_yes"],
        "temps_entrainement_s": temps_total
    }

    del modele

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return resultat

# %% CELL 63
# Partie : C
# 5. Comparer les embeddings
experiences_C = []
for arch in ['C1','C2','C3']:
    experiences_C.append(entrainer_C(f'{arch}_random',arch))
    fige = entrainer_C(f'{arch}_aravec_fige',arch,'fige')
    experiences_C.append(fige)
    experiences_C.append(entrainer_C(f'{arch}_aravec_degele',arch,'degele',depart=fige['checkpoint']))
experiences_C.append(entrainer_C('C4_charCNN','C4'))

# %% CELL 65
# Partie : C
# 6. Ablation sans positions
for embedding in ['random','fige','degele']:
    initial = sans_positions_fige['checkpoint'] if embedding=='degele' else None
    essai = entrainer_C(f'C3_sans_positions_{embedding}','C3',embedding,False,depart=initial)
    experiences_C.append(essai)
    if embedding=='fige': sans_positions_fige = essai

# %% CELL 67
# Partie : C
# 7. Comparer le déséquilibre
meilleur_C = max(experiences_C, key=lambda e:e['f1'])
print('Meilleur modèle :', meilleur_C['nom'])
for strategie in ['ponderee','surechantillonnage']:
    e = meilleur_C
    initial = None
    if e['embedding']=='degele':
        fige = entrainer_C(f'C_fige_{strategie}',e['arch'],'fige',e['positions'],strategie)
        initial = fige['checkpoint']
    entrainer_C(f'C_meilleur_{strategie}',e['arch'],e['embedding'],e['positions'],strategie,initial)

tableau_C = pd.DataFrame(resultats_C).sort_values('f1_yes',ascending=False)
display(tableau_C.round(4))
tableau_C.to_csv(OUT_C/'comparaison_dev.csv',index=False,encoding='utf-8-sig')

# %% CELL 84
# Partie : D
# 6. Entraîner D2 et garder le meilleur modèle
optimizer = torch.optim.Adam(modele_D2.parameters(), lr=1e-3)
perte = nn.BCEWithLogitsLoss()

meilleur_f1 = -1
sans_progres = 0
debut = perf_counter()

for epoch in range(8):
    modele_D2.train()

    for etats, longueurs, labels in train_loader_D:
        optimizer.zero_grad()

        logits = modele_D2(etats.to(DEVICE), longueurs)
        loss = perte(logits, labels.to(DEVICE))

        loss.backward()
        nn.utils.clip_grad_norm_(modele_D2.parameters(), 1.0)
        optimizer.step()

    scores = predire_D2()
    f1_dev = threshold_and_metrics(ydev_d, scores)["f1_yes"]

    print(f"Époque {epoch + 1} | F1-Yes dev : {f1_dev:.4f}")

    if f1_dev > meilleur_f1 + 1e-6:
        meilleur_f1 = f1_dev
        sans_progres = 0
        torch.save(modele_D2.state_dict(), OUT_D / "D2.pt")
    else:
        sans_progres += 1

    if sans_progres >= 3:
        break

if DEVICE.type == "cuda":
    torch.cuda.synchronize()

duree_D2 = perf_counter() - debut

modele_D2.load_state_dict(
    torch.load(OUT_D / "D2.pt", map_location=DEVICE, weights_only=True)
)

scores_D2 = predire_D2()
enregistrer_D("D2_BiLSTM_attention", scores_D2, duree_D2)

# %% CELL 93
# Partie : E
# 3 — Paramètres et données arabes
necessaires = [
    "train_fit", "val", "RESULTS",
    "seed_everything", "threshold_and_metrics"
]

assert all(n in globals() for n in necessaires), (
    "Exécuter les cellules de préparation C/D avant E."
)

OUT_E = RESULTS / "E"
OUT_E.mkdir(exist_ok=True)

DEVICE_E = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SEEDS_E = [42, 43, 44]
LAMBDAS_E = [0.0, 0.5, 1.0]

CFG_E = {
    "lr": 2e-5,
    "batch_size": 32,
    "micro_batch": 8,
    "min_epochs": 3,
    "max_epochs": 5,
    "patience": 2,
    "warmup": 0.10,
    "weight_decay": 0.01
}

AR_TRAIN = train_fit.reset_index(drop=True)
AR_DEV = val.reset_index(drop=True)

Y_AR = AR_TRAIN["class_label"].map(
    {"No": 0, "Yes": 1}
).to_numpy()

Y_DEV_E = AR_DEV["class_label"].map(
    {"No": 0, "Yes": 1}
).to_numpy()

assert not pd.isna(Y_AR).any()
assert not pd.isna(Y_DEV_E).any()

print("Appareil :", DEVICE_E)
print("Train arabe :", len(AR_TRAIN))
print("Dev arabe :", len(AR_DEV))

# %% CELL 101
# Partie : E
# 7 — Former les lots
def loader_E(tokens, labels, seed, shuffle=False):
    noms = list(tokens)

    dataset = TensorDataset(
        *(tokens[nom] for nom in noms),
        torch.tensor(labels, dtype=torch.float32)
    )

    loader = DataLoader(
        dataset,
        batch_size=CFG_E["batch_size"],
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed)
    )

    return noms, loader

# %% CELL 105
# Partie : E
# 9 — Définir la perte pondérée
def critere_E(y):
    poids = float(
        (y == 0).sum() / max((y == 1).sum(), 1)
    )

    return nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(poids, device=DEVICE_E)
    )

# %% CELL 107
# Partie : E
# 10 — Apprendre sur un lot
def retropropager_E(
    model, noms, batch, langue,
    facteur, critere, scaler
):
    taille = len(batch[-1])
    perte_moyenne = 0.0

    for debut in range(0, taille, CFG_E["micro_batch"]):
        fin = min(
            debut + CFG_E["micro_batch"],
            taille
        )

        inputs = {
            nom: valeurs[debut:fin].to(DEVICE_E)
            for nom, valeurs in zip(noms, batch[:-1])
        }

        labels = batch[-1][debut:fin].to(DEVICE_E)

        with torch.autocast(
            device_type=DEVICE_E.type,
            dtype=torch.float16,
            enabled=DEVICE_E.type == "cuda"
        ):
            perte = critere(
                model(inputs, langue),
                labels
            )

        poids_micro_lot = (fin - debut) / taille

        scaler.scale(
            facteur * poids_micro_lot * perte
        ).backward()

        perte_moyenne += poids_micro_lot * perte.item()

    return perte_moyenne

# %% CELL 109
# Partie : E
# 11 — Effectuer pour chaque époque
def epoque_E(
    model, donnees_ar, donnees_en, lambda_en,
    criteres, optimizer, scheduler, scaler
):
    noms_ar, loader_ar = donnees_ar

    if lambda_en > 0:
        noms_en, loader_en = donnees_en
        iter_en = iter(loader_en)

    model.train()
    somme_ar, somme_en = 0.0, 0.0

    for batch_ar in loader_ar:
        optimizer.zero_grad(set_to_none=True)

        somme_ar += retropropager_E(
            model, noms_ar, batch_ar,
            "ar", 1.0, criteres["ar"], scaler
        )

        if lambda_en > 0:
            try:
                batch_en = next(iter_en)
            except StopIteration:
                iter_en = iter(loader_en)
                batch_en = next(iter_en)

            somme_en += retropropager_E(
                model, noms_en, batch_en,
                "en", lambda_en, criteres["en"], scaler
            )

        # Mettre à jour les poids après accumulation.
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        ancien_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()

        # Ne pas avancer le scheduler si la mise à jour a été ignorée.
        if scaler.get_scale() >= ancien_scale:
            scheduler.step()

    return (
        somme_ar / len(loader_ar),
        somme_en / len(loader_ar)
    )

# %% CELL 113
# Partie : E
# 13 — Préparer un entraînement
def preparer_entrainement_E(
    partie, model_id, train_tokens,
    seed, lambda_en, en_tokens
):
    seed_everything(seed)

    model = ModeleE(
        model_id,
        deux_tetes=partie == "E2"
    ).to(DEVICE_E)

    assert all(
        p.requires_grad
        for p in model.encodeur.parameters()
    ), "L’encodeur doit être entraînable."

    donnees_ar = loader_E(
        train_tokens, Y_AR, seed, True
    )

    donnees_en = None
    criteres = {"ar": critere_E(Y_AR)}

    if lambda_en > 0:
        assert en_tokens is not None

        donnees_en = loader_E(
            en_tokens, Y_EN, seed + 1, True
        )
        criteres["en"] = critere_E(Y_EN)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CFG_E["lr"],
        weight_decay=CFG_E["weight_decay"]
    )

    total_steps = (
        len(donnees_ar[1]) * CFG_E["max_epochs"]
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=math.ceil(
            CFG_E["warmup"] * total_steps
        ),
        num_training_steps=total_steps
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=DEVICE_E.type == "cuda"
    )

    return (
        model, donnees_ar, donnees_en,
        criteres, optimizer, scheduler, scaler
    )

# %% CELL 115
# Partie : E
# 14 — Piloter l’entraînement et l’early stopping
def entrainer_E(
    nom, partie, model_id, tokenizer,
    train_tokens, dev_tokens, seed,
    lambda_en=0.0, en_tokens=None
):
    folder = OUT_E / f"{nom}_seed{seed}"
    folder.mkdir(exist_ok=True)

    (
        model, donnees_ar, donnees_en,
        criteres, optimizer, scheduler, scaler
    ) = preparer_entrainement_E(
        partie, model_id, train_tokens,
        seed, lambda_en, en_tokens
    )

    meilleur = -1.0
    attente = 0
    meilleure_epoque = 0
    historique = []

    if DEVICE_E.type == "cuda":
        torch.cuda.synchronize()
    debut = perf_counter()

    for epoch in range(1, CFG_E["max_epochs"] + 1):
        loss_ar, loss_en = epoque_E(
            model, donnees_ar, donnees_en,
            lambda_en, criteres,
            optimizer, scheduler, scaler
        )

        scores = predire_E(model, dev_tokens)
        mesures = threshold_and_metrics(Y_DEV_E, scores)

        historique.append({
            "epoch": epoch,
            "loss_ar": loss_ar,
            "loss_en": loss_en,
            **mesures
        })

        print(
            f"{nom} | graine {seed} | époque {epoch}"
            f" | F1-Yes dev {mesures['f1_yes']:.4f}"
        )

        pd.DataFrame(historique).to_csv(
            folder / "historique.csv",
            index=False
        )

        if mesures["f1_yes"] > meilleur + 1e-6:
            meilleur = mesures["f1_yes"]
            attente = 0
            meilleure_epoque = epoch

            torch.save(
                model.state_dict(),
                folder / "model.pt"
            )
        else:
            attente += 1

        if (
            epoch >= CFG_E["min_epochs"]
            and attente >= CFG_E["patience"]
        ):
            break

    if DEVICE_E.type == "cuda":
        torch.cuda.synchronize()
    duree = perf_counter() - debut

    model.load_state_dict(torch.load(
        folder / "model.pt",
        map_location=DEVICE_E,
        weights_only=True
    ))

    scores = predire_E(model, dev_tokens)

    nouveaux = sauvegarder_E(
        nom, partie, model_id, model, tokenizer,
        train_tokens, scores, seed, lambda_en,
        duree, meilleure_epoque, folder
    )

    del model, optimizer, scaler
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return nouveaux

# %% CELL 117
# Partie : E
# 15 — Lancer E1
RESULTATS_E = []

modeles_E1 = [
    (
        "E1_AraBERT",
        "aubmindlab/bert-base-arabertv02-twitter",
        64
    ),
    (
        "E1_MARBERTv2",
        "UBC-NLP/MARBERTv2",
        128
    )
]

for nom, model_id, longueur in modeles_E1:
    tok, xt, xv, _ = tokeniser_E(
        model_id, longueur
    )

    for seed in SEEDS_E:
        resultat = entrainer_E(
            nom=nom,
            partie="E1",
            model_id=model_id,
            tokenizer=tok,
            train_tokens=xt,
            dev_tokens=xv,
            seed=seed
        )

        RESULTATS_E.append(resultat)

    del tok, xt, xv
    gc.collect()

# %% CELL 119
# Partie : E
# 16 — Lancer E2
model_id = "FacebookAI/xlm-roberta-base"

tok, xt, xv, xe = tokeniser_E(
    model_id,
    longueur=128,
    anglais=True
)

for lambda_en in LAMBDAS_E:
    nom = f"E2_XLMR_lambda_{lambda_en:g}"

    for seed in SEEDS_E:
        resultat = entrainer_E(
            nom=nom,
            partie="E2",
            model_id=model_id,
            tokenizer=tok,
            train_tokens=xt,
            dev_tokens=xv,
            seed=seed,
            lambda_en=lambda_en,
            en_tokens=xe
        )

        RESULTATS_E.append(resultat)

del tok, xt, xv, xe
gc.collect()

# %% CELL 160
# Partie : F3
# F3.5 — Réentraîner avec les trois graines
resultats_train_F3 = []

for seed in SEEDS_E:
    resultat = entrainer_E(
        nom=NOM_F3,
        partie=PARTIE_F3,
        model_id=MODEL_ID_F3,
        tokenizer=tokenizer_F3,
        train_tokens=xt_F3,
        dev_tokens=xv_F3,
        seed=seed,
        lambda_en=LAMBDA_F3,
        en_tokens=xe_F3
    )

    resultats_train_F3.append(resultat)

pd.concat(
    resultats_train_F3,
    ignore_index=True
).to_csv(
    OUT_F / "F3_reentrainement_dev.csv",
    index=False,
    encoding="utf-8-sig"
)

