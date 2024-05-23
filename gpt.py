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
from pprint import pp
from sys_prompts_v4 import SYSTEM_PROMPTS


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
                 cache_dir: str = 'cache',
                 output_dir: str = 'output',
                 output_filename: str = '',
                 output_format: str = 'xlsx',
                 ):
        self.model_version: str = model_version
        self.df: pd.DataFrame = None
        self.sample_size: int = sample_size
        self.batch_size: int = batch_size
        self.sample_mode: bool = sample_mode
        self.model: str = gpt_model,
        self.response_choices: int = response_choices
        self.target_col_name: str = target_col_name
        self.index_col_name: str = index_col_name
        self.validation_column_name: str = validation_col_name
        self.validation_json_field_name: str = validation_json_field_name
        self.cache_dir: str = cache_dir
        self.cache_file: str = cache_file
        self.output_dir: str = output_dir
        self.output_filename: str = output_filename
        self.output_format: str = output_format
        self.extractions: list[tuple] = []
        self.validated: pd.DataFrame = pd.DataFrame()

    def _get_api_key(self) -> None:
        with open('OPENAI_API_KEY.txt', 'r') as key:
            return key.read()

    def _read_files(self, path: str = "data") -> pd.DataFrame:
        pth = Path(os.getcwd()) / path
        files = list(pth.glob("*.xlsx"))
        if not files:
            print("No files found in the specified directory.")
            return None
        dfs = []
        dfs_by_cols = defaultdict(list)
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
            return None
        try:
            df = pd.concat(dfs, axis=0).reset_index(drop=True)
        except Exception as e:
            print("Error concatenating dataframes.")
            print(e)
            return None
        return df

    def _build_prompts(self, summaries: list[tuple]) -> list[dict]:
        summary_prompts = [
            {
                "role": "user",
                "content": f"Here is data from the FAA Drone Sightings dataset.",
            }
        ]
        for summary in summaries:
            summary_prompts.append({"role": "user", "content": summary})
        summary_prompts.append(
            {
                "role": "user",
                "content": "Take a deep breath and solve the problem step by step. It's important you get this right.",
            }
        )
        return SYSTEM_PROMPTS + summary_prompts

    def _batch_summaries(self, summaries: list[tuple]) -> list[list[tuple]]:
        batched_summaries: list = []
        for i in range(0, len(summaries), self.batch_size):
            batched_summaries.append(summaries[i:i + self.batch_size])
        return batched_summaries

    def _extract_jsons(self, summaries: list[tuple]) -> list[dict]:
        api_key: str = self._get_api_key()
        client = OpenAI(
            api_key=api_key if api_key else os.getenv("OPENAI_API_KEY"))
        extracted: list[tuple] = []
        batched_summaries = self._batch_summaries(summaries=summaries)
        for batch in batched_summaries:
            batch_for_cache: list = []
            summaries_text = [b[1] for b in batch]
            response = client.chat.completions.create(
                model=self.model,
                response_format={"type": "json_object"},
                messages=self._build_prompts(summaries=summaries_text),
                n=self.response_choices,
            )
            json_results = [c.message.content for c in response.choices]
            results: list[tuple] = [json.loads(r) for r in json_results]
            for record_id, result_dict in zip([b[0] for b in batch], results[0]['response']):
                result_dict["ID"] = record_id
                extracted.append(result_dict)
                batch_for_cache.append(result_dict)
            self._write_to_cache(extractions=batch_for_cache, batch=True)
        return extracted

    def _prepare_summaries(self, df: pd.DataFrame) -> list[tuple]:
        if self.sample_mode:
            return [(col[0], col[1]) for col in df[[self.index_col_name, self.target_column_names]
                                                   ].dropna(how='any').sample(self.sample_size).values]
        else:
            return [(col[0], col[1]) for col in df[[self.index_col_name, self.target_column_names]
                                                   ].dropna(how='any').values]

    def _extract_data(self) -> None:
        extractions: dict = dict()
        extractions['results'] = []
        try:
            if self.cache_file:
                extractions = self._load_from_cache()
            else:
                df = self._read_files()
                summaries = self._prepare_summaries(df=df)
                # hit the api
                results: list[dict] = self._extract_jsons(summaries=summaries)
                if len(results) != len(summaries):
                    print(
                        "Error extracting JSON data. Summary and response sizes do not match."
                    )
                    return None
                for summary, result in zip(summaries, results):
                    extractions['results'].append((summary, result))
                extractions['dataframe'] = df.dropna(how='any', subset=(
                    self.index_col_name, self.target_column_names))
                self._write_to_cache(extractions)
        except FileNotFoundError as e:
            print('Cached file not found!')
        self.extractions = extractions

    def _validate_extractions(self) -> None:
        true_positive = 0
        false_positive = 0
        record_IDs = []
        manual_values = []
        gpt_values = []
        success_values = []
        try:
            df: pd.DataFrame = self.extractions['dataframe']
            df.dropna(how='any', inplace=True, subset=(
                self.index_col_name, self.target_column_names))
            for results in self.extractions['results']:
                record_ID = results[1]["ID"]
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
                self.cache_dir}/extracted_batch_{dt.now().strftime("%d_%m_%Y_%H_%M_%S_%f")}.pkl'
        else:
            filename = f'{
                self.cache_dir}/extractions_{dt.now().strftime("%d_%m_%Y_%H_%M_%S_%f")}.pkl'
        with open(filename, 'wb') as fp:
            pickle.dump(extractions, fp)

    def _load_from_cache(self) -> list[tuple]:
        with open(Path(self.cache_dir) / self.cache_file, 'rb') as fp:
            return pickle.load(fp)

    def _print_results(self, print_record_ids: list[int] = None) -> None:
        if print_record_ids:
            if print_record_ids[0] == 0:
                pp(self.extractions['results'])
            else:
                for r in print_record_ids:
                    pp(next(
                        (t[1] for t in self.extractions['results'] if t[0][0] == r), None))
                    print('\n')
        if not self.validated.empty:
            print(self.validated)
            print(f'''True Positive: {
                self.validated['success'].value_counts()[True]}''')
            print(f'''False Positive: {
                self.validated['success'].value_counts()[False]}''')

    def _write_output_file(self) -> None:
        if not self.validated.empty:
            if self.output_format == 'xlsx':
                self.validated.to_excel(
                    Path(self.output_dir) / (self.output_filename + '.xlsx'), index=False)
            else:
                #  can add additional output formats here
                pass

    def execute(self) -> None:
        self._extract_data()
        self._validate_extractions()
        # 0 for ALL records, empty for None, list of IDs for those IDs
        self._print_results(print_record_ids=[929])
        self._write_output_file()


def main() -> None:
    MODEL_VERSION = 'v4'
    GPT_MODEL = 'gpt-4o'  # gpt-4-turbo-2024-04-09
    SAMPLE_MODE = True  # False processes entire population!
    SAMPLE_SIZE = 5
    BATCH_SIZE = 1,
    RESPONSE_CHOICES = 1,
    TARGET_COL_NAME: str = 'CLEANED Summary'
    INDEX_COL_NAME: str = 'RecNum'
    VALIDATION_COL_NAME: str = 'UAS ALT'
    VALIDATION_JSON_FIELD_NAME: str = 'uas_altitude'
    CACHE_FILE: str = 'extractions_09_05_2024_01_04_12_185843.pkl'  # blank queries GPT!
    CACHE_DIR: str = f'/Users/dan/Dev/scu/InformationExtraction/cache{
        MODEL_VERSION}'
    OUTPUT_DIR: str = '/Users/dan/Dev/scu/InformationExtraction/output/gpt'
    OUTPUT_FILENAME: str = f'gtp_800-999_{MODEL_VERSION}'
    OUTPUT_FORMAT: str = 'xlsx'

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
                           cache_dir=CACHE_DIR,
                           output_dir=OUTPUT_DIR,
                           output_filename=OUTPUT_FILENAME,
                           output_format=OUTPUT_FORMAT,
                           )
    gpt.execute()


if __name__ == "__main__":
    main()
