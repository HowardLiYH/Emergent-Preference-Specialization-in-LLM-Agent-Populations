"""
Agent changelog with compaction.

Tracks behavioral evolution and summarizes when too large.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


# Compaction configuration
COMPACTION_CONFIG = {
    'trigger_threshold': 100,   # Compact when entries exceed this
    'keep_recent': 20,          # Always keep this many recent entries
    'summary_max_tokens': 500,  # Max tokens for summary
    'compaction_prompt': '''Summarize these changelog entries into a concise summary:
1. Core behavioral patterns that emerged
2. Strategies that consistently worked
3. Key failures to avoid

Entries:
{entries}

Summary (be concise, focus on actionable patterns):'''
}


@dataclass
class ChangelogEntry:
    """A single changelog entry."""
    timestamp: datetime
    event_type: str  # 'win', 'tool_upgrade', 'strategy_learned', 'compaction'
    regime: Optional[str]
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'event_type': self.event_type,
            'regime': self.regime,
            'description': self.description,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'ChangelogEntry':
        return cls(
            timestamp=datetime.fromisoformat(data['timestamp']),
            event_type=data['event_type'],
            regime=data.get('regime'),
            description=data['description'],
            metadata=data.get('metadata', {}),
        )

    def __str__(self) -> str:
        regime_str = f"[{self.regime}] " if self.regime else ""
        return f"{self.timestamp.strftime('%Y-%m-%d %H:%M')} - {regime_str}{self.description}"


class Changelog:
    """
    Changelog for tracking agent behavioral evolution.

    Features:
    - Automatic compaction when too large
    - Preserves recent entries
    - Summarizes old entries using LLM
    """

    def __init__(
        self,
        agent_id: str,
        base_path: Optional[Path] = None,
        llm_client=None,
        config: Optional[dict] = None
    ):
        """
        Initialize changelog.

        Args:
            agent_id: Agent identifier
            base_path: Base directory for storage
            llm_client: LLM client for compaction summarization
            config: Custom compaction configuration
        """
        self.agent_id = agent_id
        self.base_path = Path(base_path) if base_path else Path('.changelogs')
        self.llm_client = llm_client
        self.config = config or COMPACTION_CONFIG

        self.entries: List[ChangelogEntry] = []
        self.summaries: List[str] = []  # Compacted summaries

        # Ensure directory exists
        self.base_path.mkdir(parents=True, exist_ok=True)

        # Load existing changelog
        self._load()

    def _get_file_path(self) -> Path:
        """Get changelog file path."""
        return self.base_path / f"agent_{self.agent_id}_changelog.json"

    def _load(self) -> None:
        """Load changelog from disk."""
        file_path = self._get_file_path()

        if not file_path.exists():
            return

        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            self.entries = [
                ChangelogEntry.from_dict(e)
                for e in data.get('entries', [])
            ]
            self.summaries = data.get('summaries', [])

        except Exception as e:
            logger.error(f"Failed to load changelog for {self.agent_id}: {e}")

    def _save(self) -> None:
        """Save changelog to disk."""
        file_path = self._get_file_path()

        data = {
            'agent_id': self.agent_id,
            'entries': [e.to_dict() for e in self.entries],
            'summaries': self.summaries,
        }

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def add(
        self,
        event_type: str,
        description: str,
        regime: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> ChangelogEntry:
        """
        Add a new changelog entry.

        Args:
            event_type: Type of event
            description: Description of what happened
            regime: Optional regime context
            metadata: Optional additional data

        Returns:
            Created entry
        """
        entry = ChangelogEntry(
            timestamp=datetime.now(),
            event_type=event_type,
            regime=regime,
            description=description,
            metadata=metadata or {}
        )

        self.entries.append(entry)
        self._save()

        # Check if compaction needed
        if len(self.entries) >= self.config['trigger_threshold']:
            self._compact()

        return entry

    def record_win(self, regime: str, tool_used: str, strategy: str) -> None:
        """Record a competition win."""
        self.add(
            event_type='win',
            regime=regime,
            description=f"Won with {tool_used}: {strategy[:100]}",
            metadata={'tool': tool_used}
        )

    def record_tool_upgrade(self, new_level: int) -> None:
        """Record a tool level upgrade."""
        self.add(
            event_type='tool_upgrade',
            description=f"Upgraded to tool level L{new_level}",
            metadata={'level': new_level}
        )

    def record_strategy_learned(self, regime: str, strategy: str) -> None:
        """Record a new strategy being learned."""
        self.add(
            event_type='strategy_learned',
            regime=regime,
            description=f"Learned: {strategy[:200]}"
        )

    def _compact(self) -> None:
        """
        Compact old entries into a summary.

        Keeps recent entries and summarizes the rest.
        """
        keep_recent = self.config['keep_recent']

        if len(self.entries) <= keep_recent:
            return

        # Split entries
        old_entries = self.entries[:-keep_recent]
        recent_entries = self.entries[-keep_recent:]

        # Generate summary
        if self.llm_client:
            try:
                summary = self._generate_summary(old_entries)
            except Exception as e:
                logger.warning(f"Failed to generate summary: {e}")
                summary = self._simple_summary(old_entries)
        else:
            summary = self._simple_summary(old_entries)

        # Store summary
        self.summaries.append(summary)

        # Keep only recent entries
        self.entries = recent_entries

        # Add compaction marker
        self.entries.insert(0, ChangelogEntry(
            timestamp=datetime.now(),
            event_type='compaction',
            regime=None,
            description=f"Compacted {len(old_entries)} entries",
            metadata={'entries_compacted': len(old_entries)}
        ))

        self._save()
        logger.info(f"Compacted {len(old_entries)} entries for agent {self.agent_id}")

    def _generate_summary(self, entries: List[ChangelogEntry]) -> str:
        """Generate LLM summary of entries."""
        entries_text = "\n".join(str(e) for e in entries)

        prompt = self.config['compaction_prompt'].format(entries=entries_text)

        response = self.llm_client.generate(prompt)
        return response.text

    def _simple_summary(self, entries: List[ChangelogEntry]) -> str:
        """Generate simple summary without LLM."""
        # Count by event type and regime
        event_counts = {}
        regime_wins = {}

        for entry in entries:
            # Count events
            event_counts[entry.event_type] = event_counts.get(entry.event_type, 0) + 1

            # Count wins by regime
            if entry.event_type == 'win' and entry.regime:
                regime_wins[entry.regime] = regime_wins.get(entry.regime, 0) + 1

        # Build summary
        lines = [
            f"Summary of {len(entries)} entries:",
            f"- Events: {event_counts}",
        ]

        if regime_wins:
            top_regimes = sorted(regime_wins.items(), key=lambda x: -x[1])[:3]
            lines.append(f"- Top regimes: {dict(top_regimes)}")

        return "\n".join(lines)

    def get_recent(self, n: int = 10) -> List[ChangelogEntry]:
        """Get n most recent entries."""
        return self.entries[-n:]

    def get_all_summaries(self) -> List[str]:
        """Get all compacted summaries."""
        return self.summaries

    def get_full_history(self) -> str:
        """Get full history including summaries and entries."""
        parts = []

        # Add summaries
        for i, summary in enumerate(self.summaries):
            parts.append(f"=== Historical Summary {i+1} ===\n{summary}\n")

        # Add current entries
        if self.entries:
            parts.append("=== Recent Entries ===")
            for entry in self.entries:
                parts.append(str(entry))

        return "\n".join(parts)

    def __len__(self) -> int:
        return len(self.entries)
