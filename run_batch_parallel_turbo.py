import os
import itertools
import datetime
import hashlib
import multiprocessing
from functools import partial
from game_engine import Game, StrategyAgent, OllamaAgent, PROFILES

# Essayer d'importer tqdm, sinon utiliser une version simple
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Configuration
ROUNDS = 200  # Nombre de rounds par simulation
OUTPUT_DIR = "results"
OLLAMA_MODELS = ["qwen2.5:7b", "gemma2:9b"]
OLLAMA_TEMPERATURES = [0.7, 1.5]

# Nombre de workers parallèles - Calculé dynamiquement pour tenir dans 7h max
# Sera ajusté automatiquement dans main() pour respecter la contrainte de 7h
MAX_HOURS = 7  # Contrainte de temps maximale en heures
NUM_WORKERS = min(16, multiprocessing.cpu_count())  # Maximum initial

# Chunksize pour optimiser la communication entre processus
CHUNKSIZE = 10  # Traite les tâches par lots de 10

# Definitions
STRATEGIES = ["tit_for_tat", "random", "always_cooperate", "always_defect", "grim_trigger"]
PROFILES_KEYS = list(PROFILES.keys())
CONTEXT_OPTIONS = [True, False]

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def run_single_simulation(config_tuple):
    """
    Fonction worker optimisée pour exécuter une seule simulation.
    Version ultra-rapide avec gestion d'erreurs minimale.
    """
    config1, config2, sim_id = config_tuple
    
    try:
        # Create Agent 1
        if config1["type"] == "strategy":
            agent1 = StrategyAgent(config1["name"], config1["strategy"])
        else:
            agent1 = OllamaAgent(
                config1["name"],
                config1["model"],
                config1["profile"],
                config1["context"],
                config1["temperature"],
            )
            
        # Create Agent 2
        if config2["type"] == "strategy":
            agent2 = StrategyAgent(config2["name"], config2["strategy"])
        else:
            agent2 = OllamaAgent(
                config2["name"],
                config2["model"],
                config2["profile"],
                config2["context"],
                config2["temperature"],
            )
        
        # Filename optimisé - sans timestamp pour plus de vitesse
        name_hash = hashlib.md5(f"{agent1.name}_{agent2.name}_{sim_id}".encode()).hexdigest()[:8]
        filename = f"vs_{name_hash}_{sim_id}.parquet"
        full_path = os.path.join(OUTPUT_DIR, filename)
        
        # Run simulation (mode silencieux pour la performance)
        game = Game(agent1, agent2)
        game.run_game(ROUNDS, verbose=False)
        
        # Vérifier que l'historique n'est pas vide
        if len(game.history) == 0:
            return {
                "success": False,
                "error": "Empty history",
                "agent1": agent1.name,
                "agent2": agent2.name,
                "filename": filename
            }
        
        # Save results directement (mode silencieux)
        game.save_results(full_path, verbose=False)
        
        return {
            "success": True,
            "agent1": agent1.name,
            "agent2": agent2.name,
            "filename": filename,
            "size": len(game.history)
        }
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "agent1": config1.get("name", "Unknown"),
            "agent2": config2.get("name", "Unknown"),
            "filename": "N/A"
        }

def generate_all_combinations():
    """Génère toutes les combinaisons de configurations d'agents."""
    # Strategy Agents
    strategy_agents_configs = [{"type": "strategy", "strategy": s, "name": f"Strat_{s}"} for s in STRATEGIES]
    
    # Ollama Agents
    ollama_agents_configs = []
    for model in OLLAMA_MODELS:
        for temperature in OLLAMA_TEMPERATURES:
            for profile in PROFILES_KEYS:
                for context in CONTEXT_OPTIONS:
                    context_str = "Ctx" if context else "NoCtx"
                    model_short = model.replace(":", "").replace(".", "")
                    ollama_agents_configs.append({
                        "type": "ollama",
                        "model": model,
                        "temperature": temperature,
                        "profile": profile,
                        "context": context,
                        "name": f"O_{model_short}_{profile[:4]}_{context_str}_T{temperature}"
                    })
    
    all_configs = strategy_agents_configs + ollama_agents_configs
    
    # Generate Combinations
    combinations = list(itertools.combinations_with_replacement(all_configs, 2))
    
    # Filtrer Strategy vs Strategy et ajouter un ID unique
    tasks = []
    for sim_id, (config1, config2) in enumerate(combinations):
        # Skip Strategy vs Strategy
        if config1["type"] == "strategy" and config2["type"] == "strategy":
            continue
        tasks.append((config1, config2, sim_id))
    
    return tasks

def main():
    ensure_output_dir()
    
    print("="*60)
    print("GÉNÉRATION PARALLÈLE TURBO DES SIMULATIONS")
    print("="*60)
    print(f"Rounds par simulation: {ROUNDS}")
    print(f"Chunksize: {CHUNKSIZE}")
    print(f"Contrainte de temps: {MAX_HOURS} heures maximum")
    print(f"Répertoire de sortie: {OUTPUT_DIR}")
    print("="*60)
    
    # Générer toutes les tâches
    print("\nGénération des combinaisons...")
    tasks = generate_all_combinations()
    total_tasks = len(tasks)
    print(f"Total de simulations à exécuter: {total_tasks}")
    
    # Calculer le nombre d'appels Ollama nécessaires
    strategy_vs_ollama = sum(1 for t in tasks if t[0]["type"] == "strategy" or t[1]["type"] == "strategy")
    ollama_vs_ollama = total_tasks - strategy_vs_ollama
    total_ollama_calls = strategy_vs_ollama * ROUNDS + ollama_vs_ollama * ROUNDS * 2
    
    # Calculer le nombre de workers nécessaire pour tenir dans MAX_HOURS
    # Temps total = (total_ollama_calls * 3.5 secondes) / NUM_WORKERS
    # MAX_HOURS >= (total_ollama_calls * 3.5) / 3600 / NUM_WORKERS
    # NUM_WORKERS >= (total_ollama_calls * 3.5) / (3600 * MAX_HOURS)
    required_workers = int((total_ollama_calls * 3.5) / (3600 * MAX_HOURS)) + 1
    required_workers = max(required_workers, 1)  # Au moins 1 worker
    max_available_workers = multiprocessing.cpu_count()
    
    # Utiliser le nombre de workers calculé, limité par les CPU disponibles
    actual_workers = min(required_workers, max_available_workers, NUM_WORKERS)
    
    estimated_hours = (total_ollama_calls * 3.5) / 3600 / actual_workers
    print(f"Appels Ollama totaux: {total_ollama_calls:,}")
    print(f"Workers calculés pour tenir dans {MAX_HOURS}h: {required_workers}")
    print(f"Workers utilisés: {actual_workers}")
    print(f"Estimation de temps: ~{estimated_hours:.1f} heures (avec {actual_workers} workers)")
    
    if estimated_hours > MAX_HOURS:
        print(f"\n⚠️  ATTENTION: L'estimation ({estimated_hours:.1f}h) dépasse la limite de {MAX_HOURS}h")
        print("   Augmentez MAX_HOURS ou réduisez ROUNDS pour respecter la contrainte.")
    
    print("\nDémarrage des simulations...\n")
    
    # Exécuter en parallèle avec OPTIMISATIONS MAXIMALES
    start_time = datetime.datetime.now()
    successful = 0
    failed = 0
    errors = []
    
    with multiprocessing.Pool(processes=actual_workers) as pool:
        # Utiliser imap_unordered pour démarrer le traitement dès qu'une tâche est terminée
        # Chunksize réduit la surcharge de communication entre processus
        results = list(tqdm(
            pool.imap_unordered(run_single_simulation, tasks, chunksize=CHUNKSIZE),
            total=total_tasks,
            desc="Simulations TURBO",
            unit="sim"
        ))
        
        for result in results:
            if result["success"]:
                successful += 1
            else:
                failed += 1
                errors.append(result)
    
    end_time = datetime.datetime.now()
    duration = end_time - start_time
    
    # Résumé final
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    print(f"Simulations réussies: {successful}")
    print(f"Simulations échouées: {failed}")
    print(f"Total: {total_tasks}")
    print(f"Temps écoulé: {duration}")
    print(f"Temps moyen par simulation: {duration / total_tasks if total_tasks > 0 else 0}")
    print(f"Simulations par seconde: {total_tasks / duration.total_seconds() if duration.total_seconds() > 0 else 0:.2f}")
    
    if errors:
        print(f"\nPremières erreurs ({min(5, len(errors))} sur {len(errors)}):")
        for err in errors[:5]:
            print(f"  - {err['agent1']} vs {err['agent2']}: {err.get('error', 'Unknown')}")
    
    print("="*60)

if __name__ == "__main__":
    # Nécessaire pour Windows
    multiprocessing.freeze_support()
    main()

