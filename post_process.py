"""
Author: Dan Bright, cosmoid@tutu.io
License: GPLv3.0
Version: 1.0
First published: 20 June 2024
Description: 
    A class of methods to perform custom symbolic processing
    for the GPT information extraction pipeline (gpt.py)
"""
import csv
from pprint import pp


class PostProcess:

    def __init__(self,
                 input: list[dict],
                 expected_range: tuple = (),
                 missing_ids_file: str = '',) -> None:
        self.input = input
        self.expected_range = expected_range
        self.missing_ids_file = missing_ids_file

    def _zero_no_ac_involved(self) -> None:
        for inference in self.input:
            if inference['no_ac_involved']:
                inference['uas_altitude'] = 0

    def _remove_duplicates(self) -> None:
        print(f'Record before removal: {len(self.input)}')
        cleaned: list[dict] = []
        unique_ids = list(set(inf['record_id'] for inf in self.input))
        for rec in self.input:
            if rec['record_id'] in unique_ids:
                cleaned.append(rec)
                unique_ids.remove(rec['record_id'])
        self.input = cleaned
        print(f'Record count after removal: {len(self.input)}')

    def _identify_missing(self) -> None:
        ids: set[int] = set(i['record_id'] for i in self.input)
        expected: set[int] = set(i for i in range(
            self.expected_range[0], self.expected_range[1] + 1))
        id_list = list(ids)
        expected_list = list(expected)
        print(f'Expected ID range is {expected_list[0]} to {
              expected_list[-1]}, with {len(expected_list)} unique values.')
        print(f'Actual ID range is {id_list[0]} to {
              id_list[-1]}, with {len(id_list)} unique values.')
        missing: set = ids ^ expected
        print(f'There were {len(list(missing))} missing values.')
        self._write_csv(data=missing)

    def _write_csv(self, data) -> None:
        if self.missing_ids_file:
            ids = sorted(data)
            with open(self.missing_ids_file, 'w', newline='') as f:
                writer = csv.writer(f)
                for id in ids:
                    writer.writerow([id])

    def execute(self) -> None:
        self._zero_no_ac_involved()
        self._remove_duplicates()
        self._identify_missing()
        return self.input
