import os
import time
import textwrap
from datetime import datetime
from zoneinfo import ZoneInfo
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm
from openai import OpenAI


class LLMExperimentRunner:
    def __init__(
        self,
        llm_model: str,
        temperature: float = 0.5,
        max_workers: int = 10,
        max_retries: int = 3,
        base_delay: float = 3.0,
        persona_checkpoint_file: str = "persona_checkpoint.csv",
        experiment_checkpoint_file: str = "experiment_checkpoint.csv",
        save_every: int = 25,
        timezone: str = "Europe/Vienna",
    ):
        self.llm_model = llm_model
        self.temperature = temperature
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.persona_checkpoint_file = persona_checkpoint_file
        self.experiment_checkpoint_file = experiment_checkpoint_file
        self.save_every = save_every
        self.tz = ZoneInfo(timezone)

        load_dotenv()
        self._init_llm_client()

    def _init_llm_client(self):
        if self.llm_model == "deepseek-chat V3.2":
            api_key_env = "DEEPSEEK_API_KEY"
            self.model_name = "deepseek-chat"
            base_url = "https://api.deepseek.com"
            
        # elif self.llm_model == "qwen":
    #     # ADJUST THIS LATER
    #     api_key_env = "DEEPSEEK_API_KEY"
    #     self.model_name = "deepseek-chat"
    #     base_url = "https://api.deepseek.com"
        
    # elif self.llm_model == "llama":
    #     # ADJUST THIS LATER
    #     api_key_env = "DEEPSEEK_API_KEY"
    #     self.model_name = "deepseek-chat"
    #     base_url = "https://api.deepseek.com"
        else:
            raise ValueError(f"Unknown llm_model: {self.llm_model}")

        self.client = OpenAI(
            api_key=os.environ.get(api_key_env),
            base_url=base_url
        )

    def call_llm(self, sys_prompt: str, message: str):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": message},
            ],
            temperature=self.temperature,
            stream=False
        )

        date = datetime.fromtimestamp(response.created, tz=self.tz)
        response_text = response.choices[0].message.content

        return response_text, date

    def _generate_single_persona(self, idx, row, persona_system_prompt):
        if pd.notna(row["llm_persona"]):
            return idx, None, None

        message = "\n".join(f"{col}: {row[col]}" for col in row.index)

        for attempt in range(1, self.max_retries + 1):
            try:
                response_text, date = self.call_llm(persona_system_prompt, message)
                return idx, response_text, date

            except Exception as e:
                msg = str(e).lower()
                if "timeout" in msg or "timed out" in msg:
                    if attempt == self.max_retries:
                        return idx, f"[ERROR: timeout after {self.max_retries} retries]", None
                    time.sleep(self.base_delay * (2 ** (attempt - 1)))
                else:
                    return idx, f"[ERROR: {str(e)}]", None

    def generate_personas(self, df: pd.DataFrame, persona_system_prompt: str):
        
        if "llm_persona" not in df.columns:
            df["llm_persona"] = None
            
        if "persona_generation_date" not in df.columns:
            df["persona_generation_date"] = None

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [
                executor.submit(
                    self._generate_single_persona,
                    row.name,
                    row,
                    persona_system_prompt
                )
                for _, row in df.iterrows()
            ]

            for i, future in enumerate(
                tqdm(as_completed(futures), total=len(futures), desc="Generating personas")
            ):
                idx, response_text, date = future.result()

                if response_text is None:
                    continue

                df.at[idx, "llm_persona"] = response_text
                df.at[idx, "persona_generation_date"] = date

                if i > 0 and i % self.save_every == 0:
                    df.to_csv(self.persona_checkpoint_file, index=False)

        df.to_csv(self.persona_checkpoint_file, index=False)
        return df

            
    def _run_single_call(self, df, idx, row, experiment, col_response):
        # skip if already filled
        if pd.notna(df.at[idx, col_response]):
            return idx, None, None

        persona_system_prompt = textwrap.dedent(f"""
            Your task is to behave in the way that the Personality Profile describes:
            {row['llm_persona']}
            and respond in this format:
            {experiment.sys_instruction}
        """).strip()

        for attempt in range(1, self.max_retries + 1):
            try:
                response_text, date = self.call_llm(persona_system_prompt, experiment.prompt)
                return idx, response_text, date 

            except Exception as e:
                msg = str(e).lower()
                if "timeout" in msg or "timed out" in msg:
                    if attempt == self.max_retries:
                        return idx, f"[ERROR: timeout after {self.max_retries} retries]", None

                    time.sleep(self.base_delay * (2 ** (attempt - 1)))
                else:
                    return idx, f"[ERROR: {str(e)}]", None


    def run_experiment(self, df: pd.DataFrame, instructions):
        for experiment_dict in tqdm(instructions, desc="Experiments"):
            for experiment_name, experiment in experiment_dict.items():

                col_response = f"{self.llm_model}_experiment_{experiment_name}"
                col_date = f"{col_response}_date"

                if col_response not in df.columns:
                    df[col_response] = None
                if col_date not in df.columns:
                    df[col_date] = None


                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [
                        executor.submit(
                            self._run_single_call,
                            df, row.name, row, experiment, col_response
                        )
                        for _, row in df.iterrows()
                    ]

                    for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), leave=False)):
                        idx, response_text, date = future.result()

                        if response_text is None:
                            continue

                        df.at[idx, col_response] = response_text
                        df.at[idx, col_date] = date

                        if i > 0 and i % self.save_every == 0:
                            df.to_csv(self.experiment_checkpoint_file, index=False)

                df.to_csv(self.experiment_checkpoint_file, index=False)

        df.to_csv("Experiments_data.csv", index=False)
        return df
