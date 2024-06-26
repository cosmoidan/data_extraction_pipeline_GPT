"""
Author: Dan Bright, cosmoid@tutu.io
License: GPLv3.0
Version: 1.0
First published: 18 June 2024
Description: 
    A data extraction pipeline for GPT:
    - loads and outputs data from excel format spreadsheets (.xlsx)
    - Extracts features as described by user-engineered prompts
    - batches requests (user defined)
    - caches responses (inferences)
    - rebuilds responses from cache (if required)
    - validates extractions against hand-curated validation data
Usage:
    1) Define config parameters
    2) Define sys_prompts and user_prompts. These are separate
        python files that need to be created, named 'sys_prompts.py', 
        'user_prompts.py' and 'user_prompts_end.py'. They may also be 
        versioned, e.g., 'user_prompts_v02.py', however the import 
        statements need to be defined accordingly.
        These files contain the GPT prompts, and should be a list of dictionaries 
        containing the prompts; one prompt for each dictionary, in the form:
            * SYSTEM_PROMPTS = [{"role": "user", "content": "This is my system prompt that 
        is prepended to every request"},].
            * USER_PROMPTS = [{"role": "user", "content": "This is my user prompt that 
        prepended to every request"},].
            * USER_PROMPTS_END = [{"role": "user", "content": "This is my user end prompt 
        which is appended to every request"},].
    3) Add any custom symbolic functions for post-inference processing to 
        the 'custom_symbolic.py' file. Note, if using custom symbolic processing, 
        the CUSTOM_SYMBOLIC configuration parameter must be set to True.
    4) Run the script: python gpt.py.
Configuration Notes:
    - MODEL_VERSION: Ideally should correspond to sys_prompts version.
    - SAMPLE_MODE: False processes entire population!
    - DEFINED_SAMPLE: List of record IDs for a pre-defined sample 
        (SAMPLE_MODE has to be TRUE). Empty is random.
    - CACHE_FILE: Load final result set from this cache file. 
        Empty queries GPT API.
    - REBUILD_FROM_CACHE: If True, rebuilds final results set from 
    cache (& GPT API call is not executed). For use in case of 
    resumption after API failure. Note, input data in INPUT_DATA_DIR
    *HAS* to be identical! Also, does not work with sample!
    - BATCH_WAIT_TIME: Time to wait between batches (secs).
    - START_BATCH: Start at batch number. Set 1 to start at beginning.
    - BATCH_ATTEMPTS: Attempts to process batch in event of error.
    - PRINT_OUTPUT_OF_IDS: 0 to print results for ALL records, 
        empty for None, list of IDs to print only results for those IDs
    ADDITIONAL_REPORT_JSON_FIELD_NAMES: List of GPT inference's JSON field names
        to include in the output spreadsheet in addition to the primary 
        validation target extraction.
    - PRIMARY_DATA: An excel spreadsheet to update with the extracted values. 
    Only needs setting if EXECUTING model (not validation).
    - POST_PROCESS: Set to True if using custom (symbolic) post-inference processing
    - EXPECTED_IDX_RANGE: Tuple start and end IDs expected under the column named 
        INDEX_COL_NAME. This is used to validate successful processing of a sequence 
        of records. Leave empty if not required.
    - MISSING_IDS_FILE: Full path to csv file where missing IDs are logged. 
        This is used to validate successful processing of a sequence of records. 
        Leave blank if not required.
"""

import pickle
import pandas as pd
import os
from pathlib import Path
import json
import time
from datetime import datetime as dt
from openai import OpenAI
from typing_extensions import override
from pprint import pp
from sys_prompts_v10 import SYSTEM_PROMPTS
from user_prompts_v1 import USER_PROMPTS
from user_prompt_end_v2 import USER_PROMPTS_END
from post_process import PostProcess

MODEL_VERSION: str = 'v10'
# GPT_MODEL:str = 'gpt-4o'
GPT_MODEL: str = 'gpt-4-turbo'
SAMPLE_MODE: bool = False
DEFINED_SAMPLE: list[int] = []
SAMPLE_SIZE: int = 3
BATCH_SIZE: int = 15
BATCH_WAIT_TIME: int = 3
START_BATCH: int = 1
BATCH_ATTEMPTS: int = 5
RESPONSE_CHOICES: int = 1
TARGET_COL_NAME: str = 'CLEANED Summary'
INDEX_COL_NAME: str = 'RecNum'
VALIDATION_COL_NAME: str = 'UAS ALT'
VALIDATION_JSON_FIELD_NAME: str = 'uas_altitude'
ADDITIONAL_REPORT_JSON_FIELD_NAMES: list[str] = ['no_ac_involved']
END_SEPARATOR: str = '###'
CACHE_FILE: str = ''
REBUILD_FROM_CACHE: bool = True
BATCH_CACHE_DIR: str = f'/Users/dan/Dev/scu/InformationExtraction/cache/{
    MODEL_VERSION}/'
EXTRACTIONS_CACHE_DIR: str = BATCH_CACHE_DIR + 'extractions/'
INPUT_DATA_DIR: str = '/Users/dan/Dev/scu/InformationExtraction/data'
PRIMARY_DATA: str = '/Users/dan/Dev/scu/InformationExtraction/archive/raw_data/WIP_VERSION_3d_DB.xlsx'
OUTPUT_DIR: str = '/Users/dan/Dev/scu/InformationExtraction/output/gpt'
OUTPUT_FILENAME: str = f'model_exe_gtp4T_3d_db_{MODEL_VERSION}'
RAW_INPUT_TEXT_OUTPUT_DIR: str = '/Users/dan/Dev/scu/InformationExtraction/tmp_data'
RAW_INPUT_TEXT_OUTPUT_FILE: str = 'summaries'
RAW_INPUT_TEXT_OUTPUT_FORMAT: str = 'xlsx'
OUTPUT_FORMAT: str = 'xlsx'
EXPECTED_IDX_RANGE: tuple = (1, 17329)
VALIDATE: bool = False
PRINT_OUTPUT_OF_IDS: list[int] = []
POST_PROCESS: bool = True
LOG_FILE: str = '/Users/dan/Dev/scu/InformationExtraction/tmp_data/log.txt'
MISSING_IDS_FILE: str = '/Users/dan/Dev/scu/InformationExtraction/output/gpt/missing_ids.csv'


class GPTDataProcessor:
    def __init__(self,
                 sample_size: int = 1,
                 sample_mode: bool = True,
                 defined_sample: list[int] = [],
                 gpt_model: str = '',
                 model_version: str = 'v1',
                 batch_size: int = 1,
                 response_choices: int = 1,
                 target_col_name: str = '',
                 index_col_name: str = '',
                 validation_col_name: str = '',
                 validation_json_field_name: str = '',
                 additional_report_json_field_names: list[str] = [],
                 cache_file: str = '',
                 batch_cache_dir: str = 'cache',
                 extractions_cache_dir: str = 'cache/extractions',
                 data_dir: str = '',
                 output_dir: str = 'output',
                 output_filename: str = '',
                 output_format: str = 'xlsx',
                 end_separator: str = '###',
                 validate: bool = False,
                 print_output_of_ids: list[int] = [],
                 raw_input_text_output_dir: str = 'tmp_data',
                 raw_input_text_output_file: str = 'summaries',
                 raw_input_text_output_format: str = 'xlsx',
                 batch_wait_time: int = 90,
                 start_batch: int = 1,
                 rebuild_from_cache: bool = False,
                 batch_attempts: int = 5,
                 primary_data: str = '',
                 post_process: bool = False,
                 log_file: str = './log.txt',
                 expected_idx_range: tuple[int] = None,
                 missing_ids_file: str = ''
                 ):
        self.model_version: str = model_version
        self.original_data_df: pd.DataFrame = None
        self.df: pd.DataFrame = None
        self.primary_data: str = primary_data
        self.sample_size: int = sample_size
        self.defined_sample: list[int] = defined_sample
        self.batch_size: int = batch_size
        self.sample_mode: bool = sample_mode
        self.model: str = gpt_model
        self.response_choices: int = response_choices
        self.target_col_name: str = target_col_name
        self.index_col_name: str = index_col_name
        self.validation_column_name: str = validation_col_name
        self.validation_json_field_name: str = validation_json_field_name
        self.additional_report_json_field_names: list[str] = additional_report_json_field_names
        self.batch_cache_dir: str = batch_cache_dir
        self.extractions_cache_dir: str = extractions_cache_dir
        self.cache_file: str = cache_file
        self.output_dir: str = output_dir
        self.data_dir: str = data_dir
        self.output_filename: str = output_filename
        self.output_format: str = output_format
        self.extractions: list[tuple] = []
        self.validate: bool = validate
        self.validated: pd.DataFrame = pd.DataFrame()
        self.print_output_of_ids: list[int] = print_output_of_ids
        self.raw_input_text: list[tuple] = []
        self.raw_input_text_output_dir: str = raw_input_text_output_dir
        self.raw_input_text_output_file: str = raw_input_text_output_file
        self.raw_input_text_output_format: str = raw_input_text_output_format
        self.assistant_results: list = []
        self.end_separator: str = end_separator
        self.batch_wait_time: int = batch_wait_time
        self.start_batch: int = start_batch if start_batch != 0 else 1
        self.rebuild_from_cache: bool = rebuild_from_cache
        self.batch_attempts: int = batch_attempts
        self.post_process: bool = post_process
        self.log_file: str = log_file
        self.expected_idx_range: tuple[int] = expected_idx_range
        self.missing_ids_file: str = missing_ids_file

    def _get_api_key(self) -> None:
        with open('OPENAI_API_KEY.txt', 'r') as key:
            return key.read()

    def _read_files(self, single='') -> pd.DataFrame:
        df: pd.DataFrame = pd.DataFrame()
        if single:
            files: list[Path] = [Path(single)]
        else:
            files: list[Path] = list(Path(self.data_dir).glob("*.xlsx"))
        if not files:
            print("No files found in the specified directory.")
        dfs: list[pd.DataFrame] = []
        for file in files:
            try:
                df = pd.read_excel(file, engine="openpyxl")
            except Exception as e:
                print(f"Error reading file {file}. Skipping.")
                print(e)
                continue
            dfs.append(df)
        if not dfs:
            print("No files successfully read.")
        try:
            self.original_data_df = pd.concat(
                dfs, axis=0).reset_index(drop=True)
            self._write_output_file(mode='all_input_data')
        except Exception as e:
            print("Error concatenating DataFrames!")
            print(e)

    def _build_prompts(self, batch: list[tuple]) -> list[dict]:
        summary_prompts = USER_PROMPTS.copy()
        summaries_str = ''
        for r in batch:
            query = f'RECORD_ID: {r[0]} \n\n{r[1]} {self.end_separator}'
            summaries_str += query
        summary_prompts.append({"role": "user", "content": summaries_str})
        summary_prompts += USER_PROMPTS_END
        return SYSTEM_PROMPTS + [
            {"role": "system",
             "content": f"Each report is separated by this token: {self.end_separator}"}
        ] + summary_prompts

    def _batch_summaries(self) -> list[list[tuple]]:
        batched_summaries: list = []
        for i in range(0, len(self.raw_input_text), self.batch_size):
            batched_summaries.append(
                self.raw_input_text[i:i + self.batch_size])
        return batched_summaries

    def _execute_gpt(self) -> list[dict]:
        api_key: str = self._get_api_key()
        client = OpenAI(
            api_key=api_key if api_key else os.getenv("OPENAI_API_KEY"))
        extracted: list[tuple] = []
        batches = self._batch_summaries()
        for batch_idx, batch in enumerate(batches, 1):
            attempt_counter = 0
            batch_success = False
            while not batch_success and attempt_counter < self.batch_attempts:
                try:
                    if batch_idx >= self.start_batch:
                        print(f"Processing batch {
                            batch_idx} [Start batch was {self.start_batch}]")
                        batch_for_cache: list = []
                        response = client.chat.completions.create(
                            model=self.model,
                            response_format={"type": "json_object"},
                            messages=self._build_prompts(batch=batch),
                            n=self.response_choices,
                            max_tokens=4096,
                            temperature=0.0
                        )
                        json_results = [
                            c.message.content for c in response.choices]
                        results: list[tuple] = [
                            json.loads(r) for r in json_results]
                        for result_dict in results[0]['response']:
                            extracted.append(results[0]['response'])
                            batch_for_cache.append(result_dict)
                        if len(batch_for_cache) < len(batch):
                            pp(f'Error in this result: {result_dict}')
                            raise Exception(f"""Error processing the batch: The batch size was {
                                            self.batch_size} but there were only {len(batch_for_cache)} records returned.""")
                        else:
                            self._write_to_cache(
                                extractions=batch_for_cache, batch=True)
                            attempt_counter = 0
                            batch_success = True
                            time.sleep(self.batch_wait_time)
                    else:  #  batch previously completed as start_batch was set
                        attempt_counter = 0
                        batch_success = True
                except Exception as e:
                    print(f"""Attempt {attempt_counter + 1} failed! An error occurred while requesting batch {
                        batch_idx}. The error was: {e}""")
                    extracted.pop()
                    attempt_counter += 1
                    time.sleep(self.batch_wait_time)
            if not batch_success:
                raise Exception(
                    f'An unresolvable error occurred while processing the batch {batch_idx}')
        return extracted

    def _prepare_summaries(self) -> None:
        df = self.original_data_df
        if self.sample_mode:
            if self.defined_sample:
                filtered = df[df[self.index_col_name].isin(
                    self.defined_sample)]
                self.raw_input_text = [
                    (col[0], col[1]) for col in filtered[[self.index_col_name, self.target_col_name]].values]
            else:
                self.raw_input_text = [(col[0], col[1]) for col in df[[self.index_col_name, self.target_col_name]].dropna(
                    how='any').sample(self.sample_size).values]
        else:
            self.raw_input_text = [(col[0], col[1]) for col in df[[
                self.index_col_name, self.target_col_name]].dropna(how='any').values]

    def _extract_data(self) -> None:
        extractions: dict = dict()
        extractions['results'] = []
        try:
            if self.cache_file:
                self.extractions = self._load_extractions_from_cache()
            else:
                self._read_files()
                if not self.original_data_df.empty:
                    self._prepare_summaries()
                    if self.rebuild_from_cache:
                        results: list[dict] = self._rebuild_from_cache()
                    else:
                        results: list[dict] = self._execute_gpt()
                    if self.post_process:
                        results = PostProcess(input=results,
                                              expected_range=self.expected_idx_range,
                                              missing_ids_file=self.missing_ids_file).execute()
                    if len(results) != len(self.raw_input_text):
                        raise Exception(
                            "Warning: Summary and response sizes do not match."
                        )
                    for summary in self.raw_input_text:
                        extractions['results'].append(
                            (summary, next((d for d in results if d.get("record_id") == summary[0]), None)))
                    extractions['dataframe'] = self.original_data_df.dropna(how='any', subset=(
                        self.index_col_name, self.target_col_name))
                    self._write_to_cache(extractions, batch=False)
                    self.extractions = extractions
        except FileNotFoundError as e:
            print('Cached file not found!')
            raise Exception('Exit')
        except Exception as e:
            print(e)
            raise Exception('Exit')

    def _validate_extractions(self):
        record_IDs: list[int] = []
        manual_values: list = []
        gpt_values: list = []
        success_values: list = []
        additional_values: dict = dict()
        additional_values_dict: dict = dict()
        df: pd.DataFrame = self.extractions['dataframe']
        df.dropna(how='any', inplace=True, subset=(
            self.index_col_name, self.target_col_name))
        for results in self.extractions['results']:
            try:
                record_ID = results[1]["record_id"]
                manually_extracted = df.loc[df[self.index_col_name]
                                            == record_ID, self.validation_column_name].values[0]
                gpt_extracted = results[1][self.validation_json_field_name]
                if gpt_extracted == None:
                    gpt_extracted = 0
                record_IDs.append(record_ID)
                manual_values.append(manually_extracted)
                gpt_values.append(gpt_extracted)
                if manually_extracted == gpt_extracted:
                    success_values.append(True)
                else:
                    success_values.append(False)
                for field in self.additional_report_json_field_names:
                    if field not in additional_values.keys():
                        additional_values[field] = []
                    additional_values[field].append(
                        results[1][field] if field in results[1] else False)
                    additional_values_dict.update(
                        {field: additional_values[field]})
            except (KeyError, TypeError) as e:
                print(
                    f'An error occurred during attempted data validation of record ID {record_ID}: {e}')
        validated: pd.DataFrame = pd.DataFrame({
            'record_id': record_IDs,
            'manual': manual_values,
            'gpt': gpt_values,
            'success': success_values,
        },)
        for field, value_list in additional_values_dict.items():
            validated[field] = value_list
        validated = validated[['record_id', 'manual', 'gpt'] +
                              self.additional_report_json_field_names + ['success']]
        validated.sort_values(by='record_id', inplace=True)
        validated.reset_index(drop=True, inplace=True)
        self.validated = validated

    def _write_to_cache(self, extractions, batch=False) -> None:
        if batch:
            filename = f'{
                self.batch_cache_dir}/extracted_batch_{dt.now().strftime("%d_%m_%Y_%H_%M_%S_%f")}.pkl'
        else:
            filename = f'{
                self.extractions_cache_dir}/extractions_{dt.now().strftime("%d_%m_%Y_%H_%M_%S_%f")}.pkl'
        with open(filename, 'wb') as fp:
            pickle.dump(extractions, fp)

    def _load_extractions_from_cache(self) -> list[tuple]:
        with open(Path(self.extractions_cache_dir) / self.cache_file, 'rb') as fp:
            return pickle.load(fp)

    def _rebuild_from_cache(self) -> list[dict]:
        extracted = []
        files: list[Path] = list(Path(self.batch_cache_dir).glob("*.pkl"))
        if not files:
            print("No files found in the specified directory.")
        for f in files:
            with open(Path(f), 'rb') as fp:
                extracted += (pickle.load(fp))
        return extracted

    def _merge_dfs(self, primary: pd.DataFrame = None, to_merge: list[pd.DataFrame] = None,
                   merge_index='', primary_index='', copy_all=False) -> pd.DataFrame:
        self.additional_report_json_field_names.append(
            self.validation_json_field_name)
        for update_df in to_merge:
            update_dict: dict = update_df.set_index(
                merge_index).to_dict('index')
            for index, updates in update_dict.items():
                for col, val in updates.items():
                    if col in primary or (copy_all and
                                          col in self.additional_report_json_field_names):
                        primary.loc[primary[primary_index] == index, col] = val
        primary.sort_values(by=primary_index, inplace=True)
        return primary

    def _print_results(self) -> None:
        if self.print_output_of_ids:
            if self.print_output_of_ids[0] == 0:
                pp(self.extractions['results'])
            else:
                for r in self.print_output_of_ids:
                    pp(next(
                        (t[1] for t in self.extractions['results'] if t[0][0] == r), None))
                    print('\n')
        if self.validate and not self.validated.empty:
            pp(self.validated.head(5))
            try:
                print(f'''True Positive: {
                    self.validated['success'].value_counts()[True]}''')
            except KeyError:
                pass
            try:
                print(f'''False Positive: {
                    self.validated['success'].value_counts()[False]}''')
            except KeyError:
                pass

    def _write_log(self, message: str = '') -> None:
        with open(self.log_file, "a") as f:
            f.write(message)

    def _write_output_file(self, mode: str = None, input_df: pd.DataFrame = None) -> None:
        if mode == 'validate' and not self.validated.empty:
            if self.output_format == 'xlsx':
                self.validated.to_excel(
                    Path(self.output_dir) / (self.output_filename + '.xlsx'), index=False)
            else:
                pass  #  can add additional output formats here
        elif mode == 'summaries':
            df: pd.DataFrame = pd.DataFrame(self.raw_input_text, columns=[
                                            self.index_col_name, self.target_col_name])
            if self.raw_input_text_output_format == 'xlsx':
                df.to_excel(
                    Path(self.raw_input_text_output_dir) / (self.raw_input_text_output_file + '.xlsx'), index=False
                )
            else:
                pass  #  can add additional output formats here
        elif mode == 'all_input_data':
            output = self.original_data_df[[
                self.index_col_name, self.target_col_name]]
            output = output.sort_values(
                by=self.index_col_name, ascending=True)
            output.to_excel(
                Path(self.output_dir) / (f'input_data_for_{self.output_filename}' + '.xlsx'), index=False
            )
        elif mode == 'model_exe':
            self._read_files(single=self.primary_data)
            results: pd.DataFrame = pd.DataFrame().from_dict(
                [r[1] for r in self.extractions['results']])
            merged = self._merge_dfs(primary=self.original_data_df,
                                     to_merge=[results],
                                     merge_index='record_id',
                                     primary_index=self.index_col_name,
                                     copy_all=True)
            if self.output_format == 'xlsx':
                merged.to_excel(
                    Path(self.output_dir) / (self.output_filename + '.xlsx'), index=False)
            else:
                pass  #  can add additional output formats here
        else:
            pass  # can write more output modes here

    def execute(self) -> None:
        try:
            self._extract_data()
            if self.validate:
                self._validate_extractions()
            self._print_results()
            self._write_output_file(
                mode='validate' if self.validate else 'model_exe')
        except Exception as e:
            print(e)


def main() -> None:
    gpt = GPTDataProcessor(gpt_model=GPT_MODEL,
                           sample_mode=SAMPLE_MODE,
                           defined_sample=DEFINED_SAMPLE,
                           sample_size=SAMPLE_SIZE,
                           model_version=MODEL_VERSION,
                           batch_size=BATCH_SIZE,
                           response_choices=RESPONSE_CHOICES,
                           target_col_name=TARGET_COL_NAME,
                           index_col_name=INDEX_COL_NAME,
                           validation_col_name=VALIDATION_COL_NAME,
                           validation_json_field_name=VALIDATION_JSON_FIELD_NAME,
                           additional_report_json_field_names=ADDITIONAL_REPORT_JSON_FIELD_NAMES,
                           cache_file=CACHE_FILE,
                           batch_cache_dir=BATCH_CACHE_DIR,
                           extractions_cache_dir=EXTRACTIONS_CACHE_DIR,
                           data_dir=INPUT_DATA_DIR,
                           output_dir=OUTPUT_DIR,
                           output_filename=OUTPUT_FILENAME,
                           output_format=OUTPUT_FORMAT,
                           validate=VALIDATE,
                           print_output_of_ids=PRINT_OUTPUT_OF_IDS,
                           raw_input_text_output_file=RAW_INPUT_TEXT_OUTPUT_FILE,
                           raw_input_text_output_dir=RAW_INPUT_TEXT_OUTPUT_DIR,
                           raw_input_text_output_format=RAW_INPUT_TEXT_OUTPUT_FORMAT,
                           end_separator=END_SEPARATOR,
                           batch_wait_time=BATCH_WAIT_TIME,
                           start_batch=START_BATCH,
                           rebuild_from_cache=REBUILD_FROM_CACHE,
                           batch_attempts=BATCH_ATTEMPTS,
                           primary_data=PRIMARY_DATA,
                           post_process=POST_PROCESS,
                           log_file=LOG_FILE,
                           expected_idx_range=EXPECTED_IDX_RANGE,
                           missing_ids_file=MISSING_IDS_FILE,
                           )
    gpt.execute()


if __name__ == "__main__":
    main()
