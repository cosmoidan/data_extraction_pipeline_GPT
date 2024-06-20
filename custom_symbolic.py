"""
Author: Dan Bright, cosmoid@tutu.io
License: GPLv3.0
Version: 1.0
First published: 20 June 2024
Description: 
    A class of functions to perform custom symbolic processing
    for the GPT information extraction pipeline (gpt.py)
"""


class CustomSymbolic:

    def __init__(self) -> None:
        pass

    @staticmethod
    def zero_no_ac_involved(input: list[dict]) -> list[dict]:
        for inference in input:
            if inference['no_ac_involved']:
                inference['uas_altitude'] = 0
        return input
