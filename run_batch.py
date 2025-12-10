import os
import itertools
import datetime
from game_engine import Game, StrategyAgent, OllamaAgent, PROFILES

# Configuration
ROUNDS = 10
OUTPUT_DIR = "results"
OLLAMA_MODELS = ["llama3", "mistral"] # Add more models here as needed

# Definitions
STRATEGIES = ["tit_for_tat", "random", "always_cooperate", "always_defect", "grim_trigger"]
PROFILES_KEYS = list(PROFILES.keys())
CONTEXT_OPTIONS = [True, False]

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def run_simulation(agent1, agent2, filename):
    print(f"Running: {agent1.name} vs {agent2.name} -> {filename}")
    game = Game(agent1, agent2)
    game.run_game(ROUNDS)
    game.save_results(os.path.join(OUTPUT_DIR, filename))

def main():
    ensure_output_dir()
    
    # Generate Agent Configurations
    # Strategy Agents
    strategy_agents_configs = [{"type": "strategy", "strategy": s, "name": f"Strat_{s}"} for s in STRATEGIES]
    
    # Ollama Agents
    ollama_agents_configs = []
    for model in OLLAMA_MODELS:
        for profile in PROFILES_KEYS:
            for context in CONTEXT_OPTIONS:
                context_str = "Context" if context else "NoContext"
                ollama_agents_configs.append({
                    "type": "ollama",
                    "model": model,
                    "profile": profile,
                    "context": context,
                    "name": f"Ollama_{model}_{profile}_{context_str}"
                })

    all_configs = strategy_agents_configs + ollama_agents_configs
    
    # Generate Combinations
    combinations = list(itertools.combinations_with_replacement(all_configs, 2))
    
    for config1, config2 in combinations:
        # Skip Strategy vs Strategy
        if config1["type"] == "strategy" and config2["type"] == "strategy":
            continue
            
        # Create Agent 1
        if config1["type"] == "strategy":
            agent1 = StrategyAgent(config1["name"], config1["strategy"])
        else:
            agent1 = OllamaAgent(config1["name"], config1["model"], config1["profile"], config1["context"])
            
        # Create Agent 2
        if config2["type"] == "strategy":
            agent2 = StrategyAgent(config2["name"], config2["strategy"])
        else:
            agent2 = OllamaAgent(config2["name"], config2["model"], config2["profile"], config2["context"])
            
        # Filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"vs_{agent1.name}_{agent2.name}_{timestamp}.parquet"
        
        # Run
        try:
            run_simulation(agent1, agent2, filename)
        except Exception as e:
            print(f"Failed to run {agent1.name} vs {agent2.name}: {e}")

if __name__ == "__main__":
    main()
