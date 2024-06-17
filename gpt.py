import requests
import bs4
import pickle
import pandas as pd
import os
from pathlib import Path
import json
from collections import defaultdict
import time
from datetime import datetime as dt
from openai import OpenAI
from openai import AssistantEventHandler
from typing_extensions import override
from pprint import pp
from sys_prompts_v10 import SYSTEM_PROMPTS
from user_prompts_v1 import USER_PROMPTS
from user_prompt_end_v2 import USER_PROMPTS_END


class FAADataDownloader:

    def __init__(self, buffer_time: int = 10, data_dir: Path = None):
        if buffer_time < 10:
            raise ValueError(
                "Buffer time must be at least 10 seconds for this scraper."
            )
        self.buffer_time = buffer_time
        self.base = "https://www.faa.gov"
        self.faa_drone_report_url = (
            "https://www.faa.gov/uas/resources/public_records/uas_sightings_report"
        )
        self.data_dir: Path = data_dir

    def execute(self) -> None:
        if self.data_dir:
            self._get_file_links()
            self._download_files()
        else:
            print('Please specify a data download directory.')

    def _get_file_links(self):
        response = requests.get(self.faa_drone_report_url)
        response.raise_for_status()
        soup = bs4.BeautifulSoup(response.content, "html.parser")
        links = soup.find_all("a")
        file_links = [
            self.base + link["href"]
            for link in links
            if link.text.startswith("Reported UAS Sightings")
        ]
        return file_links

    def _download_files(self):
        file_links = self.get_file_links()
        for idx, link in enumerate(file_links):
            response = requests.get(link)
            response.raise_for_status()
            ftype = link.split(".")[-1]
            if not ftype.startswith("x"):
                print(f"Cannot download link {link}. Skipping.")
                continue
            pth = Path(os.getcwd()) / self.data_dir / \
                f"uas_sightings_report_{idx}.{ftype}"
            with open(pth, "wb") as f:
                f.write(response.content)
            print(f"Downloaded file: {pth}")
            time.sleep(self.buffer_time)
        print("Download complete.")


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
                 summaries_dir: str = 'tmp_data',
                 summaries_file: str = 'summaries',
                 summaries_format: str = 'xlsx',
                 batch_wait_time: int = 90,
                 start_batch: int = 1,
                 rebuild_from_cache: bool = False,
                 batch_attempts: int = 5,
                 ):
        self.model_version: str = model_version
        self.df: pd.DataFrame = None
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
        self.summaries: list[tuple] = []
        self.summaries_dir: str = summaries_dir
        self.summaries_file: str = summaries_file
        self.summaries_format: str = summaries_format
        self.assistant_results: list = []
        self.end_separator: str = end_separator
        self.batch_wait_time: int = batch_wait_time
        self.start_batch: int = start_batch if start_batch != 0 else 1
        self.rebuild_from_cache: bool = rebuild_from_cache
        self.batch_attempts: int = batch_attempts

    def _get_api_key(self) -> None:
        with open('OPENAI_API_KEY.txt', 'r') as key:
            return key.read()

    def _read_files(self) -> pd.DataFrame:
        df: pd.DataFrame = pd.DataFrame()
        files: list[Path] = list(Path(self.data_dir).glob("*.xlsx"))
        if not files:
            print("No files found in the specified directory.")
        dfs: list[pd.DataFrame] = []
        dfs_by_cols: defaultdict = defaultdict(list)
        for file in files:
            try:
                df = pd.read_excel(file, engine="openpyxl")
            except Exception as e:
                print(f"Error reading file {file}. Skipping.")
                print(e)
                continue
            dfs.append(df)
        for k, v in dfs_by_cols.items():
            print(k, len(v))
        if not dfs:
            print("No files successfully read.")
        try:
            df = pd.concat(dfs, axis=0).reset_index(drop=True)
        except Exception as e:
            print("Error concatenating DataFrames!")
            print(e)
        self._write_output_file(mode='all_input_data', input_df=df)
        return df

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
        for i in range(0, len(self.summaries), self.batch_size):
            batched_summaries.append(self.summaries[i:i + self.batch_size])
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
                            extracted.append(result_dict)
                            batch_for_cache.append(result_dict)
                        if len(batch_for_cache) < len(batch):
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

    def _prepare_summaries(self, df: pd.DataFrame) -> None:
        if self.sample_mode:
            if self.defined_sample:
                filtered = df[df[self.index_col_name].isin(
                    self.defined_sample)]
                self.summaries = [
                    (col[0], col[1]) for col in filtered[[self.index_col_name, self.target_col_name]].values]
            else:
                self.summaries = [(col[0], col[1]) for col in df[[self.index_col_name, self.target_col_name]].dropna(
                    how='any').sample(self.sample_size).values]
        else:
            self.summaries = [(col[0], col[1]) for col in df[[self.index_col_name, self.target_col_name]
                                                             ].dropna(how='any').values]

    def _extract_data(self) -> None:
        extractions: dict = dict()
        extractions['results'] = []
        try:
            if self.cache_file:
                extractions = self._load_extractions_from_cache()
            else:
                df = self._read_files()
                if not df.empty:
                    self._prepare_summaries(df=df)
                    if self.rebuild_from_cache:
                        results: list[dict] = self._rebuild_from_cache()
                    else:
                        results: list[dict] = self._execute_gpt()
                    if len(results) != len(self.summaries):
                        print(
                            "Error extracting JSON data. Summary and response sizes do not match."
                        )
                        return None
                    for summary in self.summaries:
                        extractions['results'].append(
                            (summary, next((d for d in results if d.get("record_id") == summary[0]), None)))
                    extractions['dataframe'] = df.dropna(how='any', subset=(
                        self.index_col_name, self.target_col_name))
                    self._write_to_cache(extractions, batch=False)
        except FileNotFoundError as e:
            print('Cached file not found!')
            raise Exception('Exit')
        except Exception as e:
            print(e)
            raise Exception('Exit')
        self.extractions = extractions

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

    def _write_output_file(self, mode: str = None, input_df: pd.DataFrame = None) -> None:
        if mode == 'validate' and not self.validated.empty:
            if self.output_format == 'xlsx':
                self.validated.to_excel(
                    Path(self.output_dir) / (self.output_filename + '.xlsx'), index=False)
            else:
                pass  #  can add additional output formats here
        elif mode == 'summaries':
            df: pd.DataFrame = pd.DataFrame(self.summaries, columns=[
                                            self.index_col_name, self.target_col_name])
            if self.summaries_format == 'xlsx':
                df.to_excel(
                    Path(self.summaries_dir) / (self.summaries_file + '.xlsx'), index=False
                )
            else:
                pass  #  can add additional output formats here
        elif mode == 'all_input_data':
            output = input_df[[self.index_col_name, self.target_col_name]]
            output = output.sort_values(
                by=self.index_col_name, ascending=True)
            output.to_excel(
                Path(self.output_dir) / (f'input_data_for_{self.output_filename}' + '.xlsx'), index=False
            )
        else:
            pass  # can write more output modes here

    def execute(self) -> None:
        try:
            self._extract_data()
            if self.validate:
                self._validate_extractions()
            self._print_results()
            self._write_output_file(
                mode='validate' if self.validate else None)
        except Exception as e:
            print(e)


def main() -> None:
    MODEL_VERSION: str = 'v10'  # ideally should correspond to sys_prompts version
    # GPT_MODEL:str = 'gpt-4o'
    GPT_MODEL: str = 'gpt-4-turbo'
    SAMPLE_MODE: bool = True  # False processes entire population!
    # list of record IDs for a pre-defined sample (SAMPLE_MODE has to be TRUE). Empty is random.
    DEFINED_SAMPLE: list[int] = []
    SAMPLE_SIZE: int = 5
    BATCH_SIZE: int = 5
    BATCH_WAIT_TIME: int = 5  # time to wait between batches (secs)
    START_BATCH: int = 81  # start at batch number. Set 1 to start at beginning!
    BATCH_ATTEMPTS: int = 5  # attempts to process batch in event of error
    RESPONSE_CHOICES: int = 1
    TARGET_COL_NAME: str = 'CLEANED Summary'
    INDEX_COL_NAME: str = 'RecNum'
    VALIDATION_COL_NAME: str = 'UAS ALT'
    VALIDATION_JSON_FIELD_NAME: str = 'uas_altitude'
    ADDITIONAL_REPORT_JSON_FIELD_NAMES: list[str] = [
        'no_ac_involved', 'multiple_events']
    END_SEPARATOR: str = '###'
    # load final result set from cache. blank queries GPT!
    CACHE_FILE: str = '/Users/dan/Dev/scu/InformationExtraction/cache/v10/extractions/extractions_15_06_2024_21_58_55_711864.pkl'
    # REBUILD_FROM_CACHE: If True, rebuilds results set from cache (& GPT API call is not executed). For use in case of resumption after API failure. Note, input data in INPUT_DATA_DIR *HAS* to be identical! Also, does not work with sample!
    REBUILD_FROM_CACHE: bool = False
    BATCH_CACHE_DIR: str = f'/Users/dan/Dev/scu/InformationExtraction/cache/{
        MODEL_VERSION}/'
    EXTRACTIONS_CACHE_DIR: str = BATCH_CACHE_DIR + 'extractions/'
    INPUT_DATA_DIR: str = '/Users/dan/Dev/scu/InformationExtraction/data'
    OUTPUT_DIR: str = '/Users/dan/Dev/scu/InformationExtraction/output/gpt'
    OUTPUT_FILENAME: str = f'gtp4T_3d_db_{MODEL_VERSION}'
    SUMMARIES_DIR: str = '/Users/dan/Dev/scu/InformationExtraction/tmp_data'
    SUMMARIES_FILE: str = 'summaries'
    SUMMARIES_FORMAT: str = 'xlsx'
    OUTPUT_FORMAT: str = 'xlsx'
    VALIDATE: bool = True
    # print_results: 0 for ALL records, empty for None, list of IDs for those IDs
    PRINT_OUTPUT_OF_IDS: list[int] = [1090]

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
                           summaries_file=SUMMARIES_FILE,
                           summaries_dir=SUMMARIES_DIR,
                           summaries_format=SUMMARIES_FORMAT,
                           end_separator=END_SEPARATOR,
                           batch_wait_time=BATCH_WAIT_TIME,
                           start_batch=START_BATCH,
                           rebuild_from_cache=REBUILD_FROM_CACHE,
                           batch_attempts=BATCH_ATTEMPTS,
                           )
    gpt.execute()


if __name__ == "__main__":
    main()
