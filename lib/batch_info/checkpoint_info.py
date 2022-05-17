from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict


@dataclass
class CheckpointInfo:
    checkpoint_info: List[Dict[str, str]] = field(default_factory=list)

    def add_info(self, batch_info: List[Dict[str, str]]) -> None:
        self.checkpoint_info += [*batch_info]

    def to_records(self) -> Dict[str, List[str]]:
        records = defaultdict(list)
        for item in self.checkpoint_info:
            for key in item:
                records[key] += [item[key]]
        return records
