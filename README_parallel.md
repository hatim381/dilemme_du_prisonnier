# Scripts de Parallélisation

Un script optimisé est disponible pour exécuter les simulations en parallèle :

## `run_batch_parallel_turbo.py` (Version TURBO - Optimisée) ⚡

Version ultra-optimisée pour une vitesse maximale.

**Utilisation :**
```bash
# Installer tqdm si nécessaire
pip install tqdm

# Exécuter
python run_batch_parallel_turbo.py
```

**Caractéristiques :**
- **Calcul automatique du nombre de workers** pour respecter la contrainte de 7h maximum
- **200 rounds** par simulation (configurable)
- Utilise `imap_unordered` pour démarrer le traitement dès qu'une tâche est terminée
- `chunksize` optimisé pour réduire la surcharge de communication
- Génération de noms de fichiers simplifiée (sans timestamp)
- Mode silencieux complet (pas de prints dans game_engine)
- **Optimisé pour tenir dans 7 heures maximum**

## Configuration

Le script est configuré pour optimiser automatiquement les performances :

```python
ROUNDS = 200                    # Nombre de rounds par simulation
MAX_HOURS = 7                   # Contrainte de temps maximale (heures)
OUTPUT_DIR = "results"          # Répertoire de sortie
```

### Ajustement automatique des workers

Le script calcule automatiquement le nombre de workers nécessaire pour respecter la contrainte de `MAX_HOURS` (7 heures par défaut). Le calcul est basé sur :
- Le nombre total d'appels Ollama nécessaires
- Une estimation de 3.5 secondes par appel Ollama
- Le nombre de CPU disponibles

**Le script ajuste automatiquement le nombre de workers pour tenir dans le temps imparti.**

## Performance

Avec ROUNDS = 200 et MAX_HOURS = 7 :
- **Temps estimé** : ≤ 7 heures (calculé automatiquement)
- **Workers** : Calculés automatiquement pour respecter la contrainte de temps
- Le script affiche une estimation précise avant de démarrer

## Résultats

Les fichiers sont sauvegardés dans le répertoire `results/` avec des noms uniques incluant :
- Hash MD5 des noms d'agents
- ID de simulation unique

## Gestion des erreurs

Les scripts continuent même si certaines simulations échouent. Un résumé des erreurs est affiché à la fin.

