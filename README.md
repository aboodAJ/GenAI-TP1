# Génération de Musique Irlandaise avec RNN

## 1. Introduction
Ce travail pratique a pour objectif de concevoir et d'entraîner un Réseau de Neurones Récurrent (RNN), spécifiquement un LSTM (Long Short-Term Memory), pour générer des mélodies de musique irlandaise au format ABC. Nous utilisons la bibliothèque PyTorch pour l'implémentation.

## 2. Analyse et Prétraitement des Données
Les données proviennent du dataset `sander-wood/irishman` et sont fournies au format JSON, vous pouvez les télécharger en utilisant le script `download_dataset.sh`. Chaque entrée contient la notation ABC d'une chanson.

### 2.1 Le Format ABC
La notation ABC utilise des caractères ASCII pour représenter les notes, les durées et les métadonnées musicales (titre, clé, mesure).
Exemple d'en-tête :
```
X:1
T:Nom de la chanson
M:4/4
K:G
```

### 2.2 Vocabulaire et Vectorisation
Pour que le modèle puisse traiter ces données textuelles, nous avons construit un vocabulaire composé de tous les caractères uniques présents dans le corpus d'entraînement.
Chaque caractère est associé à un index unique (Mapping `char` -> `int`).

- **Vectorisation** : Chaque chanson est convertie en une séquence d'entiers.
- **Input/Target** : Pour l'entraînement, nous utilisons une approche "décalage de 1" (Next Character Prediction).
    - Entrée : `séquence[0 : -1]`
    - Cible : `séquence[1 :]`

## 3. Architecture du Modèle
Nous avons choisi une architecture classique pour la modélisation de séquences de caractères.

### 3.1 Composants
1.  **Embedding Layer** : Projette chaque index de caractère dans un espace vectoriel dense de dimension 256. Cela permet de capturer des relations sémantiques entre les caractères.
2.  **LSTM Layer** : Le cœur du modèle. Avec une taille d'état caché (`hidden_size`) de 1024, il capture les dépendances temporelles à long terme, ce qui est crucial pour la musique (structure, répétitions).
3.  **Linear Layer (Dense)** : Projette la sortie du LSTM vers la taille du vocabulaire pour prédire le prochain caractère (logits).

### 3.2 Hyperparamètres
- `embedding_dim` : 256
- `hidden_size` : 1024
- `learning_rate` : 5e-3
- `batch_size` : 32 (peut être augmenté selon la VRAM)
- `epochs` : 3 (On peut l'augmenter pour des meilleurs résultats)

## 4. Entraînement
L'entraînement utilise la fonction de perte **CrossEntropyLoss** qui est adaptée aux problèmes de classification multi-classes (ici, prédire le prochain caractère parmi le vocabulaire). L'optimiseur **Adam** est utilisé pour sa convergence rapide.

Nous surveillons deux métriques :
- **Training Loss** : Pour vérifier l'apprentissage.
- **Validation Loss** : Pour vérifier la généralisation et éviter le surapprentissage (Overfitting).
Le "Early Stopping" sauvegarde le meilleur modèle basé sur la validation loss.

## 5. Génération
La génération de nouvelles musiques se fait caractère par caractère.
1. On fournit une séquence d'amorce (ex: "X:1").
2. Le modèle prédit les probabilités du prochain caractère.
3. On échantillonne le prochain caractère en utilisant une température (`temperature`) pour contrôler la créativité :
    - Température basse (< 1.0) : Plus conservateur, moins d'erreurs, mais plus répétitif.
    - Température haute (> 1.0) : Plus créatif, mais risque d'incohérence syntaxique.

## 6. Résultats et Discussion
Le output d'entrainement montre une diminution progressive de la `train_loss` et de la `val_loss`.

Et les mélodies générées respectent la structure globale (entête, mesures) mais peuvent parfois présenter des incohérences harmoniques locales.

## 7. Conclusion
Ce TP a permis de mettre en œuvre la chaîne complète de modélisation générative par RNN, du traitement de texte brut à la génération de contenu structuré. Le modèle LSTM se révèle efficace pour apprendre la syntaxe rigide du format ABC tout en capturant les motifs mélodiques.
