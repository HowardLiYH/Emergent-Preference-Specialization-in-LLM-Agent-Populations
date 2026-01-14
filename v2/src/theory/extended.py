"""
Theorem 4: Non-uniform equilibrium distribution.

Extends v1 Theorems 1-3 to handle non-uniform regime distributions.
"""

from typing import Dict, Any
import math


# =============================================================================
# THEOREM 4 (Non-Uniform Equilibrium Distribution)
# =============================================================================
#
# Under non-uniform regime distribution with:
#   - frequencies f_r
#   - rewards R_r
#   - difficulties D_r
#
# The equilibrium number of specialists in regime r is:
#
#     n_r ∝ (f_r × R_r × D_r)^(2/3)
#
# PROOF SKETCH:
#
# 1. SETUP
#    - Let γ = 0.5 (fitness sharing exponent from v1)
#    - Fitness sharing penalty: p(n) = 1/n^γ
#    - Let n_r = number of specialists in regime r
#
# 2. EXPECTED VALUE ANALYSIS
#    For an agent specializing in regime r:
#
#    EV(r) = P(regime r sampled) × P(win | sampled) × Reward
#          = f_r × (D_r / n_r) × R_r × (1/n_r^γ)  [fitness sharing]
#          = f_r × R_r × D_r / n_r^(1+γ)
#
# 3. EQUILIBRIUM CONDITION
#    At equilibrium, expected values are equal across regimes:
#
#    EV(r) = EV(s) for all regime pairs (r, s)
#
#    f_r × R_r × D_r / n_r^(1+γ) = f_s × R_s × D_s / n_s^(1+γ)
#
# 4. SOLVING FOR RATIO
#    (n_r / n_s)^(1+γ) = (f_r × R_r × D_r) / (f_s × R_s × D_s)
#
#    n_r / n_s = [(f_r × R_r × D_r) / (f_s × R_s × D_s)]^(1/(1+γ))
#
# 5. SUBSTITUTING γ = 0.5
#    1/(1+γ) = 1/1.5 = 2/3
#
#    Therefore:
#        n_r ∝ (f_r × R_r × D_r)^(2/3)
#
#    QED.
#
# =============================================================================
# NOTE ON MARKOV PROPERTY
# =============================================================================
#
# Memory introduces history dependence in ACTION selection (which tool to use),
# but LEVEL dynamics remain Markov:
#   - Next level depends only on current level + competition outcome
#   - Memory affects performance, not level transition probabilities
#
# Theorems 1-3 from v1 concern level accumulation:
#   - Theorem 1: Monotonic level increase (still holds)
#   - Theorem 2: Convergence to specialized equilibrium (still holds)
#   - Theorem 3: Stationary distribution concentration (still holds)
#
# The memory system enhances performance within levels but does not
# change the fundamental dynamics of level progression.
# =============================================================================


class Theorem4:
    """
    Theorem 4: Non-uniform equilibrium distribution.
    """

    @staticmethod
    def compute_equilibrium_distribution(
        regimes: Dict[str, Dict[str, float]],
        n_agents: int = 12,
        gamma: float = 0.5
    ) -> Dict[str, float]:
        """
        Compute theoretical equilibrium distribution.

        Args:
            regimes: Dict[regime_name -> {'frequency': f, 'reward': R, 'difficulty': D}]
            n_agents: Total number of agents
            gamma: Fitness sharing exponent

        Returns:
            Dict[regime_name -> expected_specialists]
        """
        exponent = 1 / (1 + gamma)  # = 2/3 when gamma = 0.5

        # Calculate raw proportions
        raw_proportions = {}
        for name, params in regimes.items():
            f = params.get('frequency', 0.2)
            R = params.get('reward', 1.0)
            D = params.get('difficulty', 0.5)

            value = (f * R * D) ** exponent
            raw_proportions[name] = value

        # Normalize to sum to n_agents
        total = sum(raw_proportions.values())

        distribution = {
            name: (prop / total) * n_agents
            for name, prop in raw_proportions.items()
        }

        return distribution

    @staticmethod
    def verify_equilibrium(
        observed: Dict[str, int],
        expected: Dict[str, float],
        tolerance: float = 0.15
    ) -> Dict[str, Any]:
        """
        Verify if observed distribution matches theoretical prediction.

        Args:
            observed: Observed specialist counts per regime
            expected: Expected counts from theorem
            tolerance: Relative error tolerance

        Returns:
            Verification results
        """
        errors = {}
        total_error = 0

        for regime in expected:
            obs = observed.get(regime, 0)
            exp = expected[regime]

            if exp > 0:
                error = abs(obs - exp) / exp
            else:
                error = 0 if obs == 0 else float('inf')

            errors[regime] = error
            total_error += error

        mean_error = total_error / len(expected) if expected else 0

        return {
            'matches_theorem': mean_error <= tolerance,
            'mean_relative_error': mean_error,
            'per_regime_errors': errors,
            'observed': observed,
            'expected': expected,
        }


def compute_equilibrium_distribution(
    regimes: Dict[str, Dict[str, float]],
    n_agents: int = 12,
    gamma: float = 0.5
) -> Dict[str, float]:
    """Convenience function for equilibrium computation."""
    return Theorem4.compute_equilibrium_distribution(regimes, n_agents, gamma)


# =============================================================================
# THEOREM STATEMENT (LaTeX-ready)
# =============================================================================

THEOREM_4_LATEX = r"""
\begin{theorem}[Non-Uniform Equilibrium Distribution]
\label{thm:nonuniform}
Under a non-uniform regime distribution with frequencies $f_r$, rewards $R_r$,
and difficulties $D_r$, the equilibrium number of specialists in regime $r$ satisfies:
\begin{equation}
n_r \propto (f_r \times R_r \times D_r)^{2/3}
\end{equation}
\end{theorem}

\begin{proof}
Let $\gamma = 0.5$ be the fitness sharing exponent. The expected value for a
specialist in regime $r$ is:
\begin{equation}
\text{EV}(r) = \frac{f_r \times R_r \times D_r}{n_r^{1+\gamma}}
\end{equation}

At equilibrium, $\text{EV}(r) = \text{EV}(s)$ for all regime pairs. Solving:
\begin{equation}
\frac{n_r}{n_s} = \left[\frac{f_r \times R_r \times D_r}{f_s \times R_s \times D_s}\right]^{\frac{1}{1+\gamma}}
\end{equation}

Substituting $\gamma = 0.5$, the exponent becomes $2/3$, completing the proof.
\end{proof}
"""
