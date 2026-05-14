from collections import defaultdict
from typing import Dict, Any, List
import pandas as pd
import numpy as np

class ExtractionStats:
    """Tracks the success, failure, and data quality of an extraction run."""
    def __init__(self, source_name: str):
        self.source_name = source_name
        self.total_seen = 0
        self.collected = 0
        self.skipped_reasons = defaultdict(int)
        self.failures = 0
        self.collected_rows: List[Dict[str, Any]] = []

    def record_seen(self):
        self.total_seen += 1

    def record_collected(self, row: Dict[str, Any]):
        self.collected += 1
        self.collected_rows.append(row)

    def record_skipped(self, reason: str):
        self.skipped_reasons[reason] += 1

    def record_failure(self):
        self.failures += 1
        
    def generate_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.collected_rows)

    def print_summary(self):
        print(f"\n------ {self.source_name.upper()} EXTRACTION SUMMARY ------")
        print(f"Total records seen: {self.total_seen}")
        print(f"Total records collected: {self.collected}")
        
        for reason, count in self.skipped_reasons.items():
            print(f"Skipped ({reason}):{'.' * max(0, 19 - len(reason))} {count}")
            
        print(f"Failures: {self.failures}")
        
        if self.collected > 0:
            df = self.generate_dataframe()
            print("\n--- Column Completeness ---")
            total = len(df)
            
            check_df = df.replace(r'^\s*$', np.nan, regex=True)
            filled_counts = check_df.notna().sum()
            
            for col, count in filled_counts.items():
                percentage = (count / total) * 100
                print(f"{col:25} {count:7d}/{total} ({percentage:5.1f}%)")
        print("-" * 50)
