# Architecture baselines for comparison
from .independent import IndependentTraining
from .marl_shared import MARLSharedCritic
from .tournament import TournamentSelection
from .market import MarketBasedBidding
from .cse import CompetitiveSpecialistEcosystem

__all__ = [
    'IndependentTraining',
    'MARLSharedCritic',
    'TournamentSelection',
    'MarketBasedBidding',
    'CompetitiveSpecialistEcosystem',
]
