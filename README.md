# CheckThat! 2024 — Détection d’affirmations à vérifier en arabe

**Auteure :** Sakhi Sara — Groupe 9, travail individuel  

## Objectif

Ce TP étudie la classification de tweets arabes selon leur intérêt pour la vérification des faits (*check-worthiness*), à partir de la tâche 1 de CLEF CheckThat! 2024.

- **Yes** : le tweet contient une affirmation qui mérite une vérification.
- **No** : le tweet n’est pas retenu comme affirmation à vérifier selon les annotations du corpus.

Le système ne détermine pas si une information est vraie ou fausse. Il identifie les tweets à examiner en priorité.

## Données

Source : [dépôt officiel CheckThat! 2024](https://gitlab.com/checkthat_lab/clef2024-checkthat-lab).

Archives utilisées dans `task1/data/` :

- `CT24_checkworthy_arabic.zip` pour la tâche principale.
- `CT24_checkworthy_english.zip` pour l’apprentissage auxiliaire en anglais dans E2.

Colonnes arabes : `tweet_id`, `tweet_url`, `tweet_text`, `class_label`.

| Ensemble | Effectif | Yes | No | Part de Yes |
|---|---:|---:|---:|---:|
| Train | 7 333 | 2 243 | 5 090 | 30,59 % |
| Val = dev | 1 093 | 411 | 682 | 37,60 % |
| Dev-test | 500 | 377 | 123 | 75,40 % |
| Test | 610 | 218 | 392 | 35,74 % |
| Total | 9 536 | 3 249 | 6 287 | 34,07 % |

Le total est descriptif : les quatre ensembles ne sont pas fusionnés pour entraîner les modèles.

## Protocole expérimental

- L’entraînement utilise **train** et la validation utilise **dev**.
- Le vocabulaire et les transformations apprises sont ajustés uniquement sur train.
- La sélection des configurations et l’optimisation du seuil reposent sur dev.
- Dev-test constitue une évaluation supplémentaire avec une proportion de Yes différente.
- Le test officiel est réservé aux évaluations finales de la partie F.
- Les seuils de dev sont appliqués sans modification sur dev-test et test. Les seuils optimaux calculés a posteriori servent uniquement au diagnostic.
- A à D utilisent la graine 42. E1 et E2 utilisent les graines 42, 43 et 44.
- Le sur-échantillonnage intervient uniquement dans train, après séparation des ensembles.

## Métriques

La métrique principale est le **F1 de la classe Yes**, qui combine précision et rappel. Les mesures complémentaires sont la précision et le rappel de Yes, la PR-AUC, le macro-F1 et l’accuracy.

Dans ce notebook, `pr_auc` est calculée avec `precision_recall_curve` puis `auc(recall, precision)` : il s’agit d’une intégration trapézoïdale de la courbe précision–rappel, et non de `average_precision_score`.

L’accuracy seule peut être trompeuse : sur dev, le classifieur « toujours No » atteint **62,40 % d’accuracy**, mais son **F1-Yes est nul**.

## Expériences

| Partie | Expérience |
|---|---|
| Audit | Indices superficiels, distribution des labels, doublons, contradictions et chevauchement train–dev |
| A | Règles à un indice et TF-IDF mots/caractères + régression logistique |
| B1 | Classification zero-shot NLI avec hypothèses arabes, françaises et anglaises |
| B2 | Règle : au moins une entité nommée ET un nombre hors URL/mentions |
| C | C1 BiLSTM, C2 BiLSTM + attention, C3 Mini-Transformer et C4 Char-CNN |
| D | AraBERT gelé : D1 régression logistique sur CLS/moyenne et D2 BiLSTM + attention |
| E1 | Fine-tuning d’AraBERT et de MARBERTv2 |
| E2 | Encodeur XLM-R partagé, deux têtes arabe/anglais et pondération de la perte anglaise |
| F1 | Transfert du seuil de dev, courbes F1 en fonction du seuil |
| F2 | Comparaison dev-test/test et analyse des erreurs par thème |
| F3 | Réentraînement avec nombres masqués et évaluation par sous-groupe |
| F4 | Suppression des nombres/entités sur les tweets Yes correctement détectés |
| F5 | Trois graines, bootstrap du F1 et test de McNemar avec correction de Holm |
| F6 | Annotation manuelle de 30 erreurs |
| G | Comparaison finale : qualité, temps, latence CPU et taille des poids |

### Modèles préentraînés

- **NLI** : `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`.
- **NER** : `CAMeL-Lab/bert-base-arabic-camelbert-mix-ner`.
- **AraBERT** : `aubmindlab/bert-base-arabertv02-twitter`.
- **MARBERTv2** : `UBC-NLP/MARBERTv2`.
- **XLM-R** : `FacebookAI/xlm-roberta-base`.
- **Partie C** : embeddings aléatoires ou AraVec Twitter, avec variantes gelées et dégelées.

### Prétraitement

- **TF-IDF** : normalisation arabe, n-grammes de mots 1–2 et de caractères 2–5.
- **C1–C3** : tokenisation conservant les nombres, URL, mentions, hashtags et ponctuation ; vocabulaire construit sur train, padding et troncature.
- **C4** : séquences de caractères après normalisation arabe.
- **B1** : tweet original et hypothèses, tokenizer NLI, longueur maximale de 512 tokens.
- **D/E1 AraBERT** : `ArabertPreprocessor`, tokenizer AraBERT, longueur maximale de 64 tokens.
- **E1 MARBERTv2 et E2 XLM-R** : tokenizers propres aux modèles, longueur maximale de 128 tokens dans le notebook fourni.

### Apprentissage avec l’anglais

La perte combine les deux langues :

```text
L = L_arabe + lambda × L_anglais
```

Les coefficients testés sont `lambda = 0`, `0.5` et `1`. La configuration retenue sur dev arabe est `lambda = 0.5`, avec un F1-Yes moyen de **0,5957**, contre **0,5844** pour `lambda = 0`.

### Audit et indices

Dans train, environ 66 % des tweets contiennent une URL, 75 % un hashtag et 29 % un nombre hors URL et mentions. Le lexique détecte des noms de pays dans environ 40 % des tweets et des noms d’institutions dans 6 %.

Aucun doublon exact n’est observé dans train. Après normalisation, 13 groupes de doublons sont détectés, dont un avec des labels contradictoires. L’audit ne trouve pas de chevauchement exact ou normalisé entre train et dev. Il détecte 108 paires candidates de quasi-doublons dans train avec une similarité cosinus d’au moins 0,90 sur des représentations TF-IDF de caractères.

## Principaux résultats

Résultats enregistrés dans la comparaison finale du notebook :

| Configuration | F1-Yes dev | F1-Yes dev-test | F1-Yes test | Macro-F1 test | Latence CPU (ms/tweet) | Poids (Mio) |
|---|---:|---:|---:|---:|---:|---:|
| A — TF-IDF + LR | 0,5574 | 0,8644 | 0,5365 | 0,3363 | 11,64 | 5,98 |
| B1 — NLI, hypothèses françaises | 0,5469 | 0,8584 | 0,5266 | 0,2633 | 8 842,97 | 531,79 |
| B2 — CAMeLBERT, entité + nombre | 0,3351 | 0,5261 | 0,3503 | 0,4564 | 198,28 | 413,89 |
| C — C3 sans positions, AraVec gelé, perte pondérée | 0,5694 | 0,8647 | 0,5342 | 0,3153 | 2,16 | 18,90 |
| D — AraBERT gelé + BiLSTM-attention | 0,5922 | 0,8747 | 0,5231 | 0,4491 | 174,92 | 517,41 |
| E1 — AraBERT fine-tuné | 0,6038 | 0,8419 | 0,5393 | 0,4384 | 171,71 | 515,72 |
| E2 — XLM-R arabe–anglais, lambda = 0,5 | 0,5957 | 0,8594 | 0,5451 | 0,4660 | 279,30 | 1 060,66 |

Les seuils sont choisis sur dev, sauf pour la règle fixe B2. A à D : une graine. E1 et E2 : scores moyens sur trois graines. La latence et la taille correspondent à la configuration de la graine 42 et à l’environnement de mesure utilisé. Ces latences ne constituent pas un benchmark matériel universel.

### Résultats à interpréter avec prudence

- Le NLI français optimisé prédit Yes pour **99,91 %** des tweets de dev : son F1 ne suffit pas à conclure à une bonne discrimination.
- Le score élevé sur dev-test accompagne une part de Yes de 75,40 %. Une comparaison avec test doit tenir compte du changement de distribution.
- Avec C3, AraVec gelé et perte standard, retirer les positions fait passer le F1 dev de **0,5664 à 0,5693**. Cette comparaison porte sur une graine.
- F3 compare les mêmes tweets et labels, mais avec deux traitements : original à l’entraînement/évaluation ou masquage à l’entraînement/évaluation. Sur test, le F1 passe de **0,5393 à 0,5368**.
- F4 mesure la proportion de Yes initialement bien détectés qui deviennent No après modification. Ce n’est pas une baisse du F1 sur tout le test. Sur test : **0,18 %** après suppression des nombres, **9,19 %** après suppression des entités et **9,71 %** après suppression des deux.
- McNemar avec correction de Holm ne détecte aucune différence significative entre AraBERT et MARBERTv2 sur test pour les trois graines. Une seule comparaison est significative sur dev-test, graine 42.
- F6 porte sur 30 erreurs : contexte manquant (12), bruit d’annotation présumé (10), opinion déguisée en fait (8). Ces proportions ne représentent pas le bruit de tout le corpus.

## Installation et exécution

### Environnement

Le notebook a été exécuté sur **Google Colab**. Un GPU est recommandé pour les modèles neuronaux. Les données et modèles sont téléchargés pendant l’exécution : un accès Internet est nécessaire.

Principales dépendances :

```bash
pip install numpy pandas matplotlib scipy scikit-learn torch transformers sentencepiece protobuf tqdm arabert "gensim>=4.4.0" joblib ipython
```

Les versions exactes de la session initiale ne sont pas garanties par cette commande. Si un `requirements.txt` issu de l’environnement original est disponible, le privilégier.

### Notebook principal

1. Ouvrir `Untitled15_(4)_(2) (2)(6).ipynb` dans Colab.
2. Activer un GPU pour les parties neuronales.
3. Exécuter les cellules dans leur ordre : audit, A, B, C, D, E, puis F et G.
4. Les fichiers sont écrits par défaut dans `/content/results/`.
5. Pour F6, saisir manuellement la catégorie et le commentaire de chaque erreur.
6. Sauvegarder les checkpoints et les résultats avant la fin de la session Colab.

Les cellules de présentation/import de CSV ou de téléchargement final ne sont pas nécessaires à l’entraînement.

### Limite d’exécution de G dans le fichier joint

Le notebook contient les sorties enregistrées de G, mais sa première cellule G est un correctif qui suppose une définition antérieure de `charger_modele_G`. Cette définition initiale et la préparation de `experiences_G` ne figurent pas dans le fichier joint. Pour relancer G dans une session vierge, restaurer la cellule d’initialisation et de chargement correspondante. Les résultats enregistrés ne prouvent pas que le fichier s’exécute intégralement dans une nouvelle session.

##Organisation du code
Le dossier src/ contient le code réparti en cinq fichiers :
- data.py : chargement des données.
- preprocess.py : prétraitement et préparation des textes.
- models.py : définition des modèles.
- train.py : entraînement.
- evaluate.py : évaluation et analyse des résultats.
Ces fichiers conservent les dépendances du notebook et ne s’exécutent pas séparément. Pour reproduire les expériences il faut utiliser le notebook principal.
## Fichiers générés

| Emplacement dans `results/` | Contenu |
|---|---|
| `results.csv` | Journal des configurations, graines, seuils, métriques et temps |
| `audit_*.csv` | Indices, doublons, contradictions, chevauchement et quasi-doublons |
| `A_distribution_labels.csv` | Distribution des classes par groupe |
| `B1_predictions_dev.csv` | Scores et prédictions NLI sur dev |
| `C/` | Vocabulaires, checkpoints, OOV et visualisations d’attention |
| `D/` | Modèles gelés/classifieurs et attention D2 |
| `E/` | Checkpoints E1/E2, tokenizers, configurations et historiques |
| `F/` | Évaluations finales, ablations, bootstrap, McNemar et erreurs annotées |
| `G/` | Comparaison finale, cache et mesures de coût |

Les données, poids et résultats d’une session Colab ne sont pas automatiquement inclus avec le notebook. Les chemins et noms précis des sorties sont définis dans les cellules correspondantes.

## Évaluation multilingue complémentaire — Q11

Les évaluations croisées des têtes arabe/anglaise ont été réalisées dans une exécution complémentaire d’E2, après perte des checkpoints initiaux, avec les mêmes hyperparamètres et graines. Elles doivent être accompagnées de leur notebook et de leurs résultats séparés. Le notebook principal joint ne contient pas ces évaluations croisées.

## Conclusion

AraBERT est retenu sur la validation avec un F1-Yes moyen de 0,6038. XLM-R multi-tâche obtient le meilleur F1 test observé parmi les configurations du tableau, mais avec un coût supérieur. Les faux positifs et la variation des performances entre ensembles restent des limites importantes.

Avant un déploiement, les pistes prioritaires sont la réduction des faux positifs, la validation du seuil sur des données représentatives, l’ajout de contexte et la relecture des annotations ambiguës.

## Références

- [CLEF CheckThat! 2024 — données et consignes](https://gitlab.com/checkthat_lab/clef2024-checkthat-lab)
- [Modèle NLI multilingue](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7)
- [CAMeLBERT NER](https://huggingface.co/CAMeL-Lab/bert-base-arabic-camelbert-mix-ner)
- [AraBERT Twitter](https://huggingface.co/aubmindlab/bert-base-arabertv02-twitter)
- [MARBERTv2](https://huggingface.co/UBC-NLP/MARBERTv2)
- [XLM-R](https://huggingface.co/FacebookAI/xlm-roberta-base)
- [AraVec](https://github.com/bakrianoo/aravec)
