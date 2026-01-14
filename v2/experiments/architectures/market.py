"""
Market-Based Bidding Architecture.

Alternative to fitness sharing: Explicit price mechanism.
Agents bid for tasks, prices emerge from competition.

Expected Result: Specialization via price discovery (similar to CSE).
"""

import random
from typing import Dict, Tuple
from collections import defaultdict

from .base import BaseArchitecture, BaseAgent, Regime


class MarketBasedBidding(BaseArchitecture):
    """
    Market-Based Bidding: Explicit prices instead of fitness sharing.
    
    - YES competition (auction)
    - YES price mechanism (explicit, not 1/√n)
    - YES sample sharing
    
    Expected: Similar specialization to CSE via price discovery.
    """
    
    def __init__(self, n_agents: int, regimes: Dict[str, Regime],
                 price_up: float = 1.05, price_down: float = 0.95):
        super().__init__(n_agents, regimes)
        self.name = "Market"
        
        # Per-regime prices (start at 1.0)
        self.prices: Dict[str, float] = {r: 1.0 for r in regimes}
        
        # Price adjustment factors
        self.price_up = price_up
        self.price_down = price_down
        
        # Agent balances
        self.balances: Dict[str, float] = {a.id: 10.0 for a in self.agents}
    
    def train_step(self, task: Tuple[str, str], regime: str, rng: random.Random,
                   evaluate_fn) -> Dict:
        """
        Auction-based task allocation:
        1. Agents bid based on confidence / price
        2. Highest bidder wins
        3. Winner earns reward if correct
        4. Prices adjust based on demand
        """
        question, answer = task
        current_price = self.prices[regime]
        
        # Collect bids from all agents
        bids = []
        for agent in self.agents:
            confidence = agent.get_confidence(regime)
            # Bid = confidence / price (higher confidence or lower price = higher bid)
            bid = confidence / max(current_price, 0.1)
            # Add noise for tie-breaking
            bid += rng.random() * 0.01
            bids.append((agent, bid))
        
        # Highest bidder wins the task
        winner, winning_bid = max(bids, key=lambda x: x[1])
        
        # Winner attempts task
        tool = winner.select_tool(regime, rng)
        success, tokens = evaluate_fn(winner, tool, question, answer)
        
        # Update winner
        winner.update(regime, tool, success)
        
        # Price discovery
        if success:
            # More winners → lower price (more supply)
            self.prices[regime] *= self.price_down
            self.balances[winner.id] += current_price
        else:
            # Fewer winners → higher price (less supply)
            self.prices[regime] *= self.price_up
        
        # Clamp prices
        self.prices[regime] = max(0.1, min(10.0, self.prices[regime]))
        
        self.total_tokens += tokens
        self.generations += 1
        
        return {
            'regime': regime,
            'winner': winner.id,
            'winning_bid': winning_bid,
            'tool': tool,
            'success': success,
            'price': self.prices[regime],
            'tokens': tokens,
        }
