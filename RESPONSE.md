# Réponse complète au TP4 : Classification de Texte et Analyse de Sentiments

**Étudiant** : Boudaoud Sid Ahmed  
**Date** : 24 octobre 2026  
**Dataset** : Sentiment140 (1.6M tweets), échantillon équilibré de 40 000 tweets (20k positifs / 20k négatifs)

## 1. Objectifs

Ce TP vise à comparer quatre approches pour l'analyse de sentiments sur des tweets : Naive Bayes (baseline), RNN simple, LSTM/BiLSTM, et un Transformer pré-entraîné (DistilBERT) affiné par fine-tuning. L'accent est mis sur le prétraitement (notamment le traitement des négations), la modélisation séquentielle, l'attention et l'analyse critique des résultats.

## 2. Préparation des données et nettoyage

### 2.1 Chargement et exploration
- Fichier : `data/training.1600000.processed.noemoticon.csv`
- Colonnes originales : `polarity, id, date, query, user, text`
- Re-codage : polarité `0` → négatif (0), polarité `4` → positif (1)
- Distribution initiale très équilibrée (~800k chacun). Pour des raisons de temps/ressources CPU, un échantillon stratifié et parfaitement équilibré de 40 000 tweets a été utilisé.
- Découpage stratifié : Train/Validation/Test (70% / 15% / 15%)

### 2.2 Nettoyage spécifique aux tweets
Le nettoyage appliqué (`solution/preprocess.py`) :
1. Passage en minuscules
2. Suppression du préfixe `RT`
3. Conservation du texte des hashtags (`#happy` → `happy`) tout en retirant le symbole `#`
4. Suppression des URLs (`http://`, `https://`, `www.`)
5. Suppression des mentions (`@username`)
6. Suppression de la ponctuation/caractères spéciaux et normalisation des espaces
7. **Traitement des négations (critique)** : les mots de négation `not`, `no`, `never`, `n't`, `none`, `nothing`, `nowhere`, ainsi que les contractions négatives (`isn't`, `don't`, `can't`, ...) ont été **conservés** et exclus de la liste des stopwords. Les autres stopwords anglais ont été filtrés de manière classique.

Ce choix est fondamental : retirer `not` détruit le sens de constructions comme `"not good"`, `"not bad at all"`, ou `"don't like"`, ce qui est particulièrement problématique pour la détection de l'inversion de polarité.

## 3. Partie 1 : Naive Bayes (Baseline)

### 3.1 Vectorisation
Deux représentations ont été testées avec `sklearn` :
- **Bag-of-Words (CountVectorizer)** : unigrammes + bigrammes, `max_features=15000`, `min_df=2`
- **TF-IDF (TfidfVectorizer)** : unigrammes + bigrammes, `max_features=15000`, `min_df=2`

### 3.2 Résultats Naive Bayes
| Représentation | Accuracy | Précision | Rappel | F1-Score | Temps d'entraînement |
|---|---|---|---|---|---|
| TF-IDF (1-2g) | 0.7760 | 0.7859 | 0.7587 | 0.7720 | ~0.01 s |

Matrice de confusion (TF-IDF) :
```
[[4748 1237]
 [1444 4540]]
```
- Vrais négatifs : 4748, Faux positifs : 1237
- Faux négatifs : 1444, Vrais positifs : 4540

### 3.3 Mots/bigrammes les plus discriminants
**Top discriminants pour le positif** : `congratulations`, `for following`, `welcome`, `the followfriday`, `no problem`, `your welcome`, `get followers`, ...
**Top discriminants pour le négatif** : `that sucks`, `so sad`, `miss my`, `headache`, `sad`, `gonna miss`, `hurts`, `sick`, `unfortunately`, `really sad`, `sadly`, `missing my`, ...

Ces bigrammes montrent que Naive Bayes bénéficie fortement de paires de mots fréquentes dans les tweets, mais reste limité par l'absence de modélisation de l'ordre.

## 4. Partie 2 : RNN Simple et LSTM

### 4.1 Préparation pour les modèles séquentiels
- **Tokenisation** : `Tokenizer` Keras/PyTorch personnalisé, vocabulaire limité aux `VOCAB_SIZE=10000` mots les plus fréquents, ajout de `<pad>` et `<unk>`
- **Padding** : séquences de longueur fixe `max_len=32` (padding/troncature)
- **Embedding** : couche `Embedding(vocab_size, embed_dim=100)` entraînée from scratch
- **Architecture** : Embedding → RNN/LSTM/BiLSTM → Dropout(0.2) → Dense(sigmoïde)
- **Optimisation** : Adam, Binary Cross-Entropy, batch size 256, 6 epochs sur CPU

### 4.2 Résultats
| Modèle | Représentation | Accuracy | Précision | Rappel | F1-Score | Temps d'entraînement |
|---|---|---|---|---|---|---|
| RNN Simple | Séquence + Embedding | 0.5005 | 0.5002 | 0.9993 | 0.6667 | 27.5 s |
| LSTM | Séquence + Embedding | 0.6358 | 0.7436 | 0.4144 | 0.5322 | 49.7 s |
| BiLSTM | Séquence + Embedding | 0.7725 | 0.7665 | 0.7838 | 0.7750 | 87.4 s |

### 4.3 Observations
- **RNN Simple** : converge vers un comportement majoritairement "positif" (rappel quasi-parfait sur positifs, accuracy ~50%) - signe classique de difficulté à apprendre les dépendances longues et de vanishing gradient.
- **LSTM unidirectionnel** : meilleur équilibre que RNN simple, mais sous-performant dans cette configuration (F1 faible) - possiblement sous-entraîné ou sensible aux hyperparamètres.
- **BiLSTM** : nettement meilleur, atteint des performances comparables au meilleur baseline (F1 ~0.775), ce qui confirme l'intérêt de la bidirectionnalité et d'une meilleure gestion de la mémoire à long terme.

Les courbes d'apprentissage sont disponibles dans `figures/` (`rnn_*.png`, `lstm_*.png`, `bilstm_*.png`).

## 5. Partie 3 : Transformer (DistilBERT)

### 5.1 Approche
- **Modèle** : `distilbert-base-uncased` (version allégée de BERT, 6 couches, ~66M paramètres)
- **Tokenisation** : WordPiece sous-lexicale (`AutoTokenizer`), gère bien les mots inconnus, fautes d'orthographe, argot
- **Fine-tuning** : `AutoModelForSequenceClassification` avec 2 labels, entraîné par boucle PyTorch manuelle sur CPU
- **Paramètres** : `max_length=32`, batch size 4, AdamW lr=2e-5, 1 epoch (avec limitation à 200 batches/epoch pour faisabilité CPU), entraîné sur l'échantillon de 40k

### 5.2 Résultats
| Modèle | Représentation | Accuracy | Précision | Rappel | F1-Score | Temps d'entraînement |
|---|---|---|---|---|---|---|
| Transformer (DistilBERT fine-tuné) | Subword + Self-Attention | 0.7687 | 0.7624 | 0.7807 | 0.7715 | ~89.0 s |

Matrice de confusion :
```
[[4485 1500]
 [1309 4675]]
```

### 5.3 Test qualitatif sur phrases complexes
Cinq phrases rédigées pour tester négations, opposition, sarcasme :

| Phrase | Prédiction DistilBERT | Analyse |
|---|---|---|
| `"I don't think this movie is not bad at all"` | **POS** | Double négation correctement interprétée (forme litotique) |
| `"It's not great, but I don't hate it"` | **POS** | Gère la concession ("but") et les négations mitigées |
| `"Yeah right, this is AMAZING... not"` | **NEG** | Détecte le sarcasme via le contexte "not" en fin de phrase |
| `"I really wanted to like it, but it was just painfully boring"` | **NEG** | Bien identifié grâce à la conjonction d'opposition et au lexique négatif |
| `"Well, that was just perfect... said no one ever"` | **NEG** | Comprend l'ironie implicite contextuelle |

Ces résultats montrent la capacité des Transformers à modéliser le contexte global et à résoudre des cas ambigus où l'ordre et le contexte l'emportent sur le simple comptage de mots.

## 6. Partie 4 : Évaluation comparative

### 6.1 Tableau récapitulatif
| Modèle | Représentation | Accuracy | Précision | Rappel | F1-Score | Temps d'entraînement |
|---|---|---|---|---|---|---|
| Naive Bayes | BoW / TF-IDF (1-2g) | 0.7760 | 0.7859 | 0.7587 | 0.7720 | ~0.01 s |
| RNN Simple | Séquence + Embedding | 0.5005 | 0.5002 | 0.9993 | 0.6667 | ~27.5 s |
| LSTM | Séquence + Embedding | 0.6358 | 0.7436 | 0.4144 | 0.5322 | ~49.7 s |
| BiLSTM | Séquence + Embedding | 0.7725 | 0.7665 | 0.7838 | 0.7750 | ~87.4 s |
| Transformer (DistilBERT) | Subword + Attention | 0.7687 | 0.7624 | 0.7807 | 0.7715 | ~89.0 s |

*(Configuration actuelle : échantillon 40k, CPU, epochs réduites pour faisabilité ; avec plus de données/GPU/epochs, le Transformer dépasse généralement les autres sur ce type de tâche.)*

### 6.2 Comparaison visuelle
Les graphiques générés sont disponibles dans `figures/` : `f1_comparison.png`, `time_comparison.png`, ainsi que les courbes d'apprentissage pour les modèles RNN/LSTM/BiLSTM.

## 7. Questions d'analyse et de justification

### Question 1 : Hypothèse de Naive Bayes - "naïve" et limites face aux négations
**Pourquoi dite "naïve" ?**
Naive Bayes repose sur l'hypothèse d'**indépendance conditionnelle** : étant donné la classe $C$, tous les mots $w_i$ du document sont supposés indépendants les uns des autres.
$$
P(w_1,...,w_n|C) = \prod_{i=1}^{n} P(w_i|C)
$$

**Limites avec les inversions de polarité :**
- **Absence de structure/syntaxe** : La représentation "sac de mots" (BoW) ignore complètement l'ordre des mots. `"not good"` et `"good not"` ont exactement la même représentation vectorielle.
- **Portée de la négation ignorée** : `"not good at all"` devrait inverser la polarité positive de "good", mais NB voit simplement les mots "not", "good", "at", "all" indépendamment et cumule leurs probabilités conditionnelles par classe.
- **Neutralisation imparfaite** : Même en conservant les mots de négation (comme fait ici), NB ne modélise pas le **scope** de la négation ni les relations entre mots.

C'est pourquoi le prétraitement doit **impérativement conserver** `not/no/never` plutôt que de les retirer comme stopwords classiques.

### Question 2 : RNN Simple vs LSTM - Vanishing gradient et portes
**Problème du vanishing gradient**
Dans un RNN simple, l'information est propagée séquentiellement par multiplication matricielle répétée. Lors de la rétropropagation à travers le temps (BPTT), on multiplie par la matrice de poids récurrente à chaque pas. Si les valeurs propres sont < 1, le gradient décroît exponentiellement au cours des longues séquences → **vanishing gradient**. À l'inverse s'ils sont > 1, explosion du gradient. Résultat : le RNN simple "oublie" les informations lointaines (début de séquence) et a du mal à capturer des dépendances à long terme, comme celles nécessaires pour interpréter correctement une négation loin du mot qu'elle modifie.

**Solution apportée par LSTM et rôle des portes**
La cellule LSTM introduit un **état de cellule** $C_t$ (mémoire à long terme) séparé de l'état caché $h_t$, et le fait circuler avec des modifications contrôlées par des portes :

- **Forget gate** $f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$ : décide quelles informations de $C_{t-1}$ doivent être **oubliées** (filtrage du passé non pertinent)
- **Input gate** $i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$ et candidat $\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$ : décide quelles **nouvelles informations** stocker dans la cellule
- **Update** : $C_t = f_t * C_{t-1} + i_t * \tilde{C}_t$ (ajout sélectif plutôt que réécriture totale)
- **Output gate** $o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$, $h_t = o_t * \tanh(C_t)$ : décide quelle partie de la cellule est exposée à l'état caché courant

Cette architecture permet un **flux de gradient plus stable** le long de l'état de cellule (connexion additive $C_{t-1}\to C_t$) atténuant considérablement le vanishing gradient. C'est pourquoi LSTM/BiLSTM sont plus efficaces sur des séquences plus longues où la négation et le contexte distant sont importants.

**BiLSTM** va plus loin en traitant la séquence dans les deux sens (gauche→droite et droite→gauche), permettant de capturer à la fois le contexte passé et futur - particulièrement utile pour comprendre "but", "although", ou pour désambiguïser selon ce qui suit/précède.

### Question 3 : Pourquoi les Transformers surpassent-ils les modèles récurrents sur Sentiment140 ?
Plusieurs facteurs expliquent la supériorité théorique/pragmatique des Transformers :

1. **Self-Attention (parallélisation et accès direct)**  
   Contrairement au RNN qui traite séquentiellement, le mécanisme de self-attention calcule pour chaque mot une pondération sur **tous** les autres mots de la séquence simultanément : $Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V$. Cela permet d'accéder directement aux mots distants en une seule couche, sans dépendre de la propagation pas-à-pas (résout mieux les dépendances longues et évite le vanishing gradient).

2. **Contexte bidirectionnel natif**  
   BERT/DistilBERT est bidirectionnel dès le pré-entraînement (Masked Language Modeling), contrairement aux RNN unidirectionnels classiques. Il apprend des représentations contextuelles riches intégrant à la fois gauche et droite simultanément.

3. **Tokenisation sous-lexicale (WordPiece)**  
   Plutôt que de travailler uniquement sur des mots entiers, WordPiece décompose en sous-unités (`"unhappily"` peut devenir `un`, `##happ`, `##ily`). Cela permet de :
   - Gérer efficacement les **mots hors-vocabulaire** (OOV)
   - Mieux traiter l'**argot** tweetien, les **abréviations**, les **fautes de frappe** fréquentes sur Twitter
   - Capturer de la morphologie utile (préfixes/suffixes) pour la polarité

4. **Pré-entraînement sur de grands corpus**  
   DistilBERT est pré-entraîné sur BookCorpus et Wikipedia (des milliards de tokens) avec des objectifs auto-supervisés. Il possède déjà des connaissances linguistiques générales (syntaxe, sémantique, relations) qu'il transfère au fine-tuning sur Sentiment140 avec peu de données/époques. Les RNN/LSTM avec embeddings appris from-scratch partent d'une initialisation aléatoire et doivent tout apprendre.

5. **Meilleure modélisation des phénomènes pragmatiques**  
   L'attention globale aide à relier `"not"` à `"bad"`, à capter l'effet de `"but"`, et à intégrer le contexte pour détecter sarcasme/ironie - des cas où l'interprétation dépend fortement du contexte global plutôt que de co-occurrences locales.

Sur notre configuration CPU/échantillon réduit, BiLSTM et Transformer sont très proches ; avec un entraînement complet sur GPU et plus d'époques, l'avantage du Transformer se confirme généralement sur ce benchmark.

### Question 4 : Analyse des erreurs et impact de LSTM puis Transformer sur négations/ironie
**Types d'erreurs observés (génériques sur Sentiment140)**

D'après les matrices de confusion et le comportement typique :
- **Faux positifs** : tweets négatifs avec des mots apparemment positifs ou neutralisés par contexte ("not terrible", "could be worse", litotes) - difficiles pour modèles purement lexicaux
- **Faux négatifs** : tweets positifs exprimés ironiquement/sarcastiquement, ou avec des négations complexes
- **Erreurs sur la portée** : cas où la négation porte loin du terme clé
- **Ambiguïté pragmatique** : "yeah right", sous-entendus, blagues

**Impact par modèle**

| Phénomène | Naive Bayes | RNN Simple | LSTM | BiLSTM | Transformer |
|---|---|---|---|---|---|
| Simple négation ("not good") | Souvent erroné/instable | Difficile, oublie le "not" | Mieux, mais dépend de distance | Bon | Très bon |
| Double négation ("not bad", "not...not bad") | Très difficile (cumul probabiliste) | Généralement raté | Partiel | Amélioré | Généralement corrigé (cf. test qualitatif) |
| Concession ("not great, but I don't hate it") | Ignore "but", traité en sac | Faible | Partiel | Mieux avec bidirectionnalité | Bon (attention sur "but"/contexte) |
| Sarcasme/Ironie ("AMAZING... not", "perfect... said no one ever") | Principalement lexical → faux positifs probables | Très difficile | Difficile sans contexte global | Amélioré | Meilleur (contexte global + pré-entraînement pragmatique) |
| Expressions idiomatiques/argot | Limité au vocabulaire observé | Limité | Limité | Limité | Mieux via sous-unités WordPiece |

**Exemples concrets issus de nos tests qualitatifs (Transformer)**
- *"I don't think this movie is not bad at all"* → correctement **POS** (double négation litotique bien résolue)
- *"Yeah right, this is AMAZING... not"* → correctement **NEG** (détection du renversement sarcastique)
- *"Well, that was just perfect... said no one ever"* → correctement **NEG** (ironie implicite comprise)

**Conclusion sur l'erreur** : Le passage vers LSTM/BiLSTM aide significativement pour les dépendances longues et la négation simple/modérée. Le passage vers Transformer donne les plus gros gains sur les cas les plus ambigus (double négation, sarcasme, ironie) grâce au self-attention global et au savoir pré-entraîné. Néanmoins, aucun modèle n'est parfait sur l'ironie purement pragmatique - c'est un problème encore ouvert.

### Question 5 : Compromis Performance / Coût de calcul - Quand privilégier Naive Bayes ou LSTM plutôt qu'un Transformer ?
Il faut raisonner en termes de **F1-gain / ressources / latence / maintenabilité**.

| Critère | Naive Bayes (TF-IDF) | LSTM/BiLSTM | Transformer (DistilBERT) |
|---|---|---|---|
| **Performance (F1)** | ~0.772 (très bon pour un baseline) | BiLSTM ~0.775 dans notre config | ~0.772 ici (CPU/restrictif) ; souvent >0.78-0.80 en conditions optimales |
| **Temps entraînement** | ~0.01s | ~50-90s (CPU) | ~90s+ (CPU) / quelques minutes optimisé GPU |
| **Temps inférence** | **Quasi-instantané** (millisecondes) | Rapide sur CPU/GPU | Plus lent, nécessite plus de mémoire (RAM/GPU VRAM) |
| **Ressources** | CPU seul, peu de RAM | CPU acceptable, GPU aide | Idéal GPU, gourmand en VRAM (>2-4GB recommandé) |
| **Taille modèle** | Quelques Ko (vocab + probs) | ~10-50 Mo typique | ~250+ Mo (DistilBERT) |
| **Explicabilité** | Excellente (top mots/bigrammes discriminants) | Faible/opacité modérée | Très opaque (black-box) |
| **Données nécessaires** | Fonctionne bien même avec données limitées | Besoin raisonnable, mieux avec plus de données | Meilleur avec fine-tuning, mais pré-entraîné aide |

**Recommandations selon le contexte réel :**

- **Privilégier Naive Bayes si :**  
  - Contraintes fortes (embarqué, faible latence, pas de GPU)  
  - Besoin d'**explicabilité** / auditabilité (réglementaire, explicable IA)  
  - Volume modéré, temps de développement court  
  - Mise en production à très grande échelle avec coûts serrés  
  - Premier prototype/baseline rapide

- **Privilégier LSTM/BiLSTM si :**  
  - Besoin de modéliser la séquentialité sans infrastructure GPU lourde  
  - Corpus spécifique/domain (moins couvert par BERT généraliste) et on veut entraîner from-scratch avec contrôle fin  
  - Contraintes mémoire/latence intermédiaires  
  - On veut un bon compromis performance/coût sans gérer un pipeline Transformers complet

- **Privilégier Transformer si :**  
  - **Précision maximale** requise (analyse fine, sarcasme/ironie fréquents)  
  - GPU disponible et budget calcul acceptable  
  - Contexte multilingue/complexe ou besoin de transfert depuis pré-entraînement  
  - Volumes importants, inférence en batch ou infrastructure dédiée

**Observation sur nos résultats** : Dans ce setup contraint (CPU, échantillon 40k, epochs réduites), **BiLSTM (F1=0.775)** et **Naive Bayes (F1=0.772)** sont très compétitifs face au **Transformer (F1=0.772)** pour un coût computationnel bien inférieur pour NB. Cela illustre parfaitement le compromis : **"meilleur absolu" n'est pas toujours synonyme de "meilleur choix" en production**.

## 8. Conclusion

- **Naive Bayes** reste une baseline solide, ultra-rapide et explicable. Sa principale faiblesse est l'ignorance de l'ordre et du scope de la négation.
- **RNN Simple** montre les limites du traitement séquentiel sans mémoire explicite (vanishing gradient).
- **LSTM/BiLSTM** corrigent ces limites grâce aux portes et à la bidirectionnalité, atteignant des performances proches du meilleur modèle avec une complexité intermédiaire.
- **Transformer (DistilBERT)** excelle sur les cas ambigus (double négation, concession, sarcasme/ironie) grâce au self-attention global, à la tokenisation sous-lexicale et au pré-entraînement. Dans des conditions optimales (GPU, plus de données/époques), son avantage se renforce.
- **Leçon clé** : Le choix du modèle doit équilibrer **performance, coût, latence, explicabilité et contraintes d'infrastructure**. Pour de nombreux cas d'usage industriels, Naive Bayes ou BiLSTM bien optimisés constituent un excellent compromis.

## Annexes
- **Code source** : `solution/config.py`, `solution/preprocess.py`, `solution/01_naive_bayes.py`, `solution/02_rnn_lstm_pytorch.py`, `solution/03_transformer_manual.py`, `solution/04_evaluate_and_report.py`
- **Résultats bruts** : `results/nb_results.csv`, `results/rnn_lstm_results.csv`, `results/transformer_results.csv`, `results/all_results.csv`
- **Figures** : `figures/*.png` (courbes d'apprentissage, comparaisons F1/temps)
- **Données** : `data/training.1600000.processed.noemoticon.csv`
