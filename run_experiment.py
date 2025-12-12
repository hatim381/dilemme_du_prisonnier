import argparse
import os
from game_engine import Game, StrategyAgent, OllamaAgent

def main():
    parser = argparse.ArgumentParser(description="Run Prisoner's Dilemma Experiment")
    
    # Agent 1 Configuration
    parser.add_argument("--agent1_type", type=str, default="strategy", choices=["strategy", "ollama"], help="Type of Agent 1")
    parser.add_argument("--agent1_name", type=str, default="TitForTat", help="Name of Agent 1")
    parser.add_argument("--agent1_strategy", type=str, default="tit_for_tat", help="Strategy for Agent 1 (if type is strategy)")
    parser.add_argument("--agent1_model", type=str, default="qwen2.5:7b", help="Ollama model for Agent 1")
    parser.add_argument("--agent1_profile", type=str, default="default", help="Behavioral profile for Agent 1 (e.g., cooperative, grudger)")
    parser.add_argument("--agent1_temperature", type=float, default=0.7, help="Temperature for Agent 1 (Ollama)")

    # Agent 2 Configuration
    parser.add_argument("--agent2_type", type=str, default="strategy", choices=["strategy", "ollama"], help="Type of Agent 2")
    parser.add_argument("--agent2_name", type=str, default="Random", help="Name of Agent 2")
    parser.add_argument("--agent2_strategy", type=str, default="random", help="Strategy for Agent 2 (if type is strategy)")
    parser.add_argument("--agent2_model", type=str, default="gemma2:9b", help="Ollama model for Agent 2")
    parser.add_argument("--agent2_profile", type=str, default="default", help="Behavioral profile for Agent 2 (e.g., cooperative, grudger)")
    parser.add_argument("--agent2_temperature", type=float, default=1.5, help="Temperature for Agent 2 (Ollama)")

    # Game Configuration
    parser.add_argument("--rounds", type=int, default=10, help="Number of rounds to play")
    parser.add_argument("--output", type=str, default="experiment_results.parquet", help="Output filename for results")

    args = parser.parse_args()

    # Initialize Agent 1
    if args.agent1_type == "strategy":
        agent1 = StrategyAgent(args.agent1_name, args.agent1_strategy)
    else:
        agent1 = OllamaAgent(args.agent1_name, args.agent1_model, args.agent1_profile, temperature=args.agent1_temperature)

    # Initialize Agent 2
    if args.agent2_type == "strategy":
        agent2 = StrategyAgent(args.agent2_name, args.agent2_strategy)
    else:
        agent2 = OllamaAgent(args.agent2_name, args.agent2_model, args.agent2_profile, temperature=args.agent2_temperature)

    # Run Game
    game = Game(agent1, agent2)
    game.run_game(args.rounds)
    
    # Save Results
    game.save_results(args.output)

if __name__ == "__main__":
    main()
