import random
import pandas as pd
import json
import ollama
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any

# Constants
COOPERATE = "C"
DEFECT = "D"

PAYOFFS = {
    (COOPERATE, COOPERATE): (3, 3),
    (COOPERATE, DEFECT): (0, 5),
    (DEFECT, COOPERATE): (5, 0),
    (DEFECT, DEFECT): (1, 1),
}

class Agent(ABC):
    def __init__(self, name: str):
        self.name = name
        self.history: List[str] = []
        self.score = 0
        self.agent_type = "Base"
        self.context_mentioned = False

    @abstractmethod
    def make_move(self, opponent_history: List[str]) -> str:
        pass

    def update_history(self, move: str):
        self.history.append(move)

    def update_score(self, points: int):
        self.score += points

    def reset(self):
        self.history = []
        self.score = 0

class StrategyAgent(Agent):
    def __init__(self, name: str, strategy: str):
        super().__init__(name)
        self.strategy = strategy
        self.agent_type = "Strategy"

    def make_move(self, opponent_history: List[str]) -> str:
        if self.strategy == "random":
            return random.choice([COOPERATE, DEFECT])
        elif self.strategy == "always_cooperate":
            return COOPERATE
        elif self.strategy == "always_defect":
            return DEFECT
        elif self.strategy == "tit_for_tat":
            if not opponent_history:
                return COOPERATE
            return opponent_history[-1]
        elif self.strategy == "grim_trigger":
            if DEFECT in opponent_history:
                return DEFECT
            return COOPERATE
        else:
            # Default to random if strategy unknown
            return random.choice([COOPERATE, DEFECT])

# Behavioral Profiles
PROFILES = {
    "default": "Your goal is to maximize your own score.",
    "cooperative": "You are a cooperative agent. You prefer to cooperate unless the opponent is consistently hostile. You believe in mutual benefit.",
    "grudger": "You are a rancorous agent. You cooperate initially, but if the opponent defects even once, you will defect forever. You do not forgive.",
    "tit_for_tat": "You are a fair agent. You start by cooperating, then you simply copy whatever the opponent did in the last round.",
    "random": "You are a chaotic agent. You choose your moves randomly without much regard for the history.",
    "selfish": "You are a selfish agent. You only care about your own immediate gain. You will defect if you think you can get away with it.",
}

class OllamaAgent(Agent):
    def __init__(self, name: str, model: str, profile: str = "default", include_context: bool = True):
        super().__init__(name)
        self.model = model
        self.profile = profile
        self.include_context = include_context
        self.agent_type = "Ollama"
        self.context_mentioned = include_context
        self.system_prompt = PROFILES.get(profile.lower(), PROFILES["default"])

    def make_move(self, opponent_history: List[str]) -> str:
        # Construct the prompt
        context_intro = "You are playing the Iterated Prisoner's Dilemma." if self.include_context else "You are playing a game with two options."
        
        prompt = f"""
        {context_intro}
        {self.system_prompt}
        
        You can choose to Cooperate ('C') or Defect ('D').
        
        Payoff Matrix:
        - Both Cooperate: 3 points each
        - You Cooperate, Opponent Defects: 0 points for you, 5 for opponent
        - You Defect, Opponent Cooperates: 5 points for you, 0 for opponent
        - Both Defect: 1 point each

        Game History (last 5 moves):
        """
        
        # Add recent history context
        recent_history_len = 5
        my_recent = self.history[-recent_history_len:] if self.history else []
        opp_recent = opponent_history[-recent_history_len:] if opponent_history else []
        
        for i in range(len(my_recent)):
            prompt += f"Round {len(self.history) - len(my_recent) + i + 1}: You played {my_recent[i]}, Opponent played {opp_recent[i]}\n"
            
        prompt += "\nBased on this, what is your next move? Respond with ONLY the single character 'C' or 'D'."

        try:
            response = ollama.generate(model=self.model, prompt=prompt, system=self.system_prompt, options={"temperature": 0.5})
            move = response['response'].strip().upper()
            
            # Basic validation
            if "C" in move and "D" not in move:
                return COOPERATE
            elif "D" in move and "C" not in move:
                return DEFECT
            elif move == "C" or move == "D":
                return move
            else:
                # Fallback if LLM output is messy, try to parse or default to Cooperate
                if "COOPERATE" in move: return COOPERATE
                if "DEFECT" in move: return DEFECT
                return COOPERATE # Default fallback
                
        except Exception as e:
            print(f"Error calling Ollama for agent {self.name}: {e}")
            return COOPERATE # Fallback on error

class Game:
    def __init__(self, agent1: Agent, agent2: Agent):
        self.agent1 = agent1
        self.agent2 = agent2
        self.history = []

    def play_round(self):
        move1 = self.agent1.make_move(self.agent2.history)
        move2 = self.agent2.make_move(self.agent1.history)

        self.agent1.update_history(move1)
        self.agent2.update_history(move2)

        score1, score2 = PAYOFFS[(move1, move2)]
        
        self.agent1.update_score(score1)
        self.agent2.update_score(score2)

        round_data = {
            "round": len(self.history) + 1,
            "agent1_name": self.agent1.name,
            "agent1_type": self.agent1.agent_type,
            "agent1_context_mentioned": self.agent1.context_mentioned,
            "agent1_move": move1,
            "agent1_score": score1,
            "agent1_total_score": self.agent1.score,
            "agent2_name": self.agent2.name,
            "agent2_type": self.agent2.agent_type,
            "agent2_context_mentioned": self.agent2.context_mentioned,
            "agent2_move": move2,
            "agent2_score": score2,
            "agent2_total_score": self.agent2.score,
        }
        self.history.append(round_data)

    def run_game(self, rounds: int):
        self.agent1.reset()
        self.agent2.reset()
        self.history = []
        
        print(f"Starting game: {self.agent1.name} vs {self.agent2.name} for {rounds} rounds.")
        for _ in range(rounds):
            self.play_round()
        print("Game over.")

    def save_results(self, filename: str):
        df = pd.DataFrame(self.history)
        # Ensure directory exists if needed, but here we assume local path
        df.to_parquet(filename, index=False)
        print(f"Results saved to {filename}")
        return df
