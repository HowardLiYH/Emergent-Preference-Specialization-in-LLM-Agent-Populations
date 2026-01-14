"""
Lightweight Router for Deployment.

Trains a simple router from competition outcomes to route
incoming tasks to appropriate specialists.
"""

import random
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import json


@dataclass
class RoutingExample:
    """A single routing training example."""
    task_text: str
    regime: str
    specialist_id: str
    confidence: float


@dataclass
class SpecialistProfile:
    """Profile of a specialist agent."""
    agent_id: str
    specialty: str
    win_rate: float
    total_wins: int
    tool_preferences: Dict[str, float] = field(default_factory=dict)
    sample_prompts: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'agent_id': self.agent_id,
            'specialty': self.specialty,
            'win_rate': self.win_rate,
            'total_wins': self.total_wins,
            'tool_preferences': self.tool_preferences,
        }


class CompetitionRouter:
    """
    Router trained from competition outcomes.
    
    Uses simple keyword matching + win statistics to route tasks.
    In production, could be replaced with a fine-tuned classifier.
    """
    
    def __init__(self):
        self.specialists: Dict[str, SpecialistProfile] = {}
        self.regime_keywords: Dict[str, List[str]] = {
            'pure_qa': ['what', 'who', 'where', 'when', 'capital', 'wrote', 'symbol'],
            'code_math': ['calculate', 'compute', 'multiply', 'divide', 'code', 'function'],
            'chart_analysis': ['chart', 'graph', 'plot', 'visualize', 'trend'],
            'document_qa': ['document', 'file', 'read', 'extract', 'passage'],
            'realtime_data': ['current', 'today', 'now', 'latest', 'price', 'weather'],
        }
        self.routing_history: List[RoutingExample] = []
    
    def train_from_competition(self, agents: List, regimes: Dict):
        """
        Train router from competition outcomes.
        
        Extracts specialist profiles from trained agents.
        """
        for agent in agents:
            specialty = agent.get_specialty() if hasattr(agent, 'get_specialty') else None
            
            if specialty:
                # Calculate win rate
                total_wins = agent.wins if hasattr(agent, 'wins') else 0
                total_attempts = sum(agent.total_attempts.values()) if hasattr(agent, 'total_attempts') else 1
                win_rate = total_wins / max(total_attempts, 1)
                
                # Extract tool preferences from beliefs
                tool_prefs = {}
                if hasattr(agent, 'beliefs') and specialty in agent.beliefs:
                    for tool, belief in agent.beliefs[specialty].items():
                        tool_prefs[tool] = belief.mean() if hasattr(belief, 'mean') else 0.5
                
                profile = SpecialistProfile(
                    agent_id=agent.id if hasattr(agent, 'id') else str(agent),
                    specialty=specialty,
                    win_rate=win_rate,
                    total_wins=total_wins,
                    tool_preferences=tool_prefs,
                )
                
                self.specialists[specialty] = profile
    
    def route(self, task_text: str) -> Tuple[str, float]:
        """
        Route a task to the best specialist.
        
        Returns:
            Tuple of (regime, confidence)
        """
        task_lower = task_text.lower()
        
        # Score each regime based on keyword matches
        scores = {}
        for regime, keywords in self.regime_keywords.items():
            score = sum(1 for kw in keywords if kw in task_lower)
            scores[regime] = score
        
        # Add specialist win rate as a factor
        for regime, profile in self.specialists.items():
            if regime in scores:
                scores[regime] += profile.win_rate * 2
        
        if not scores or max(scores.values()) == 0:
            # Default to most common regime
            return 'pure_qa', 0.3
        
        best_regime = max(scores, key=scores.get)
        confidence = min(1.0, scores[best_regime] / 5)  # Normalize to 0-1
        
        return best_regime, confidence
    
    def get_specialist(self, regime: str) -> Optional[SpecialistProfile]:
        """Get the specialist for a regime."""
        return self.specialists.get(regime)
    
    def export_profiles(self, path: str):
        """Export specialist profiles to JSON."""
        profiles = {k: v.to_dict() for k, v in self.specialists.items()}
        with open(path, 'w') as f:
            json.dump(profiles, f, indent=2)
    
    def import_profiles(self, path: str):
        """Import specialist profiles from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        for regime, profile_data in data.items():
            self.specialists[regime] = SpecialistProfile(**profile_data)


def extract_specialist_profiles(agents: List) -> Dict[str, SpecialistProfile]:
    """
    Extract specialist profiles from trained agents for deployment.
    
    Returns a mapping from regime to specialist profile.
    """
    profiles = {}
    
    for agent in agents:
        specialty = agent.get_specialty() if hasattr(agent, 'get_specialty') else None
        
        if not specialty:
            continue
        
        # Only keep best specialist per regime
        total_wins = agent.wins if hasattr(agent, 'wins') else 0
        
        if specialty not in profiles or total_wins > profiles[specialty].total_wins:
            # Extract tool preferences
            tool_prefs = {}
            if hasattr(agent, 'beliefs') and specialty in agent.beliefs:
                for tool, belief in agent.beliefs[specialty].items():
                    tool_prefs[tool] = belief.mean() if hasattr(belief, 'mean') else 0.5
            
            # Extract sample prompts from memory if available
            sample_prompts = []
            if hasattr(agent, 'memories'):
                for mem in agent.memories[-3:]:  # Last 3 memories
                    if hasattr(mem, 'content'):
                        sample_prompts.append(mem.content)
            
            profiles[specialty] = SpecialistProfile(
                agent_id=agent.id if hasattr(agent, 'id') else str(agent),
                specialty=specialty,
                win_rate=total_wins / max(1, sum(getattr(agent, 'total_attempts', {}).values())),
                total_wins=total_wins,
                tool_preferences=tool_prefs,
                sample_prompts=sample_prompts,
            )
    
    return profiles
