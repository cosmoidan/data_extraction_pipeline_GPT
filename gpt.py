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
from sys_prompts_v6 import SYSTEM_PROMPTS
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
                 gpt_model: str = '',
                 model_version: str = 'v1',
                 batch_size: int = 1,
                 response_choices: int = 1,
                 target_col_name: str = '',
                 index_col_name: str = '',
                 validation_col_name: str = '',
                 validation_json_field_name: str = '',
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
                 ):
        self.model_version: str = model_version
        self.df: pd.DataFrame = None
        self.sample_size: int = sample_size
        self.batch_size: int = batch_size
        self.sample_mode: bool = sample_mode
        self.model: str = gpt_model
        self.response_choices: int = response_choices
        self.target_col_name: str = target_col_name
        self.index_col_name: str = index_col_name
        self.validation_column_name: str = validation_col_name
        self.validation_json_field_name: str = validation_json_field_name
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
        return df

    def _build_prompts(self, batch: list[tuple]) -> list[dict]:
        summary_prompts = USER_PROMPTS
        summaries_str = f'{self.end_separator}'.join(
            [f'RECORD_ID: {s[0]} \n\n{s[1]}' for s in batch]) + self.end_separator
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

    def _format_instructions(self) -> str:
        return f"{'. '.join([p['content'].rstrip('.') for p in SYSTEM_PROMPTS])}."

    def _format_user_prompts(self) -> str:
        return f"{'. '.join([p['content'].rstrip('.') for p in USER_PROMPTS])}."

    def _execute_gpt(self) -> list[dict]:
        api_key: str = self._get_api_key()
        client = OpenAI(
            api_key=api_key if api_key else os.getenv("OPENAI_API_KEY"))
        extracted: list[tuple] = []
        batched_summaries = self._batch_summaries()
        for batch_idx, batch in enumerate(batched_summaries, 1):
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
                )
                json_results = [c.message.content for c in response.choices]
                try:
                    results: list[tuple] = [
                        json.loads(r) for r in json_results]
                    for result_dict in results[0]['response']:
                        extracted.append(result_dict)
                        batch_for_cache.append(result_dict)
                    if len(batch_for_cache) < self.batch_size:
                        raise Exception(f"""Error processing the batch: The batch size was {
                                        self.batch_size} but there were only {len(batch_for_cache)} records returned.""")
                    self._write_to_cache(
                        extractions=batch_for_cache, batch=True)
                except Exception as e:
                    print(f"""An error occurred while requesting batch {
                        batch_idx}. The error was: {e}""")
                    raise Exception(f"""Error processing batch: {batch_idx}.
                                    Error details: {e}.
                                    Batch content: {batch}""")
                time.sleep(self.batch_wait_time)
        return extracted

    def _prepare_summaries(self, df: pd.DataFrame) -> None:
        if self.sample_mode:
            self.summaries = [(col[0], col[1]) for col in df[[self.index_col_name, self.target_col_name]
                                                             ].dropna(how='any').sample(self.sample_size).values]
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
                        # hit the api
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
        self.extractions = extractions

    def _validate_extractions(self):
        true_positive = 0
        false_positive = 0
        record_IDs = []
        manual_values = []
        gpt_values = []
        success_values = []
        try:
            df: pd.DataFrame = self.extractions['dataframe']
            df.dropna(how='any', inplace=True, subset=(
                self.index_col_name, self.target_col_name))
            for results in self.extractions['results']:
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
                    true_positive += 1
                    success_values.append(True)
                else:
                    false_positive += 1
                    success_values.append(False)
            validated: pd.DataFrame = pd.DataFrame({
                'record_id': record_IDs,
                'manual': manual_values,
                'gpt': gpt_values,
                'success': success_values,
            },)
            validated.sort_values(by='record_id', inplace=True)
            validated.reset_index(drop=True, inplace=True)
            self.validated = validated
        except KeyError as e:
            print(f'An error occurred during attempted data validation: {e}')

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

    def _write_output_file(self, mode=None) -> None:
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
        else:
            pass  # can write non-validation output here

    def execute(self) -> None:
        self._extract_data()
        if self.validate:
            self._validate_extractions()
        self._print_results()
        self._write_output_file(
            mode='validate' if self.validate else None)


def main() -> None:
    MODEL_VERSION = 'v7'
    GPT_MODEL = 'gpt-4o'  # gpt-4-turbo-2024-04-09
    SAMPLE_MODE = True  # False processes entire population!
    SAMPLE_SIZE = 15
    BATCH_SIZE = 5
    BATCH_WAIT_TIME = 5  # time to wait between batches (secs)
    START_BATCH = 1  # start at batch number. Set 1 to start at beginning!
    RESPONSE_CHOICES = 1
    TARGET_COL_NAME: str = 'CLEANED Summary'
    INDEX_COL_NAME: str = 'RecNum'
    VALIDATION_COL_NAME: str = 'UAS ALT'
    VALIDATION_JSON_FIELD_NAME: str = 'uas_altitude'
    END_SEPARATOR: str = '###'
    CACHE_FILE: str = ''  # load final result set from cache. blank queries GPT!
    # REBUILD_FROM_CACHE: If True, rebuilds results set from cache. For use in case of resumption after API failure. Note, input data in INPUT_DATA_DIR *HAS* to be identical! Also, does not work with sample!
    REBUILD_FROM_CACHE: bool = False
    BATCH_CACHE_DIR: str = f'/Users/dan/Dev/scu/InformationExtraction/cache/{
        MODEL_VERSION}/'
    EXTRACTIONS_CACHE_DIR: str = BATCH_CACHE_DIR + 'extractions/'
    INPUT_DATA_DIR: str = '/Users/dan/Dev/scu/InformationExtraction/data'
    OUTPUT_DIR: str = '/Users/dan/Dev/scu/InformationExtraction/output/gpt'
    OUTPUT_FILENAME: str = f'gtp4o_3d_db_{MODEL_VERSION}'
    SUMMARIES_DIR: str = '/Users/dan/Dev/scu/InformationExtraction/tmp_data'
    SUMMARIES_FILE: str = 'summaries'
    SUMMARIES_FORMAT: str = 'xlsx'
    OUTPUT_FORMAT: str = 'xlsx'
    VALIDATE: bool = True
    # print_results: 0 for ALL records, empty for None, list of IDs for those IDs
    PRINT_OUTPUT_OF_IDS = []

    gpt = GPTDataProcessor(gpt_model=GPT_MODEL,
                           sample_mode=SAMPLE_MODE,
                           sample_size=SAMPLE_SIZE,
                           model_version=MODEL_VERSION,
                           batch_size=BATCH_SIZE,
                           response_choices=RESPONSE_CHOICES,
                           target_col_name=TARGET_COL_NAME,
                           index_col_name=INDEX_COL_NAME,
                           validation_col_name=VALIDATION_COL_NAME,
                           validation_json_field_name=VALIDATION_JSON_FIELD_NAME,
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
                           )
    gpt.execute()


if __name__ == "__main__":
    main()
