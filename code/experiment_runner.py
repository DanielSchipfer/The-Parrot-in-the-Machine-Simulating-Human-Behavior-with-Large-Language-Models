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
from typing import List, Dict, Literal, Optional
from dataclasses import dataclass
import logging
from pathlib import Path

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "llm_experiments.log"
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

@dataclass(frozen=True) # make it immutable
class Experiment:
    """
    Immutable container defining a single LLM experiment.

    Attributes
    ----------
    prompt : str
        The user prompt that will be sent to the LLM.
    sys_instruction : str
        The system-level instruction specifying how the model should respond
        (e.g., output format, constraints).
    """
    prompt: str
    sys_instruction: str

class LLMExperimentRunner:
    def __init__(
        self,
        llm_model: str,
        thinking: Literal["enabled", "disabled"],
        reasoning_effort: Optional[Literal["high", "max"]],
        temperature: float = 1.0,
        max_workers: int = 10,
        max_retries: int = 3,
        base_delay: float = 3.0,
        save_every: int = 25,
        timezone: str = "Europe/Vienna",
        enable_logging: bool = False
    ):
        """
        Initialize the experiment runner and LLM client.

        Parameters
        ----------
        llm_model : str
            Identifier of the LLM backend to use 
            Choose between "deepseek-v4-pro, "deepseek-v4-flash".
        temperature : float, optional
            Sampling temperature for the LLM, by default 1.0.
        max_workers : int, optional
            Maximum number of concurrent threads, by default 10.
        max_retries : int, optional
            Maximum number of retries per LLM call, by default 3.
        base_delay : float, optional
            Base delay (seconds) for exponential backoff, by default 3.0.
        save_every : int, optional
            Save checkpoint CSV every N completed calls, by default 25.
        timezone : str, optional
            Timezone used to localize LLM response timestamps.
        enable_logging : bool, optional
            Whether to enable detailed logging for each LLM call.
        """
        self.llm_model = llm_model
        self.temperature = temperature
        self.thinking = thinking
        self.reasoning_effort = reasoning_effort
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.save_every = save_every
        self.tz = ZoneInfo(timezone)
        self.enable_logging = enable_logging
        self.data_dir = Path("generated_data")
        self.data_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._configure_logging()
        load_dotenv()
        self._init_llm_client()

    def _configure_logging(self):
        """Attach file logging only when requested; otherwise suppress it."""
        self.logger.propagate = False
        for handler in list(self.logger.handlers):
            self.logger.removeHandler(handler)
            handler.close()

        if self.enable_logging:
            self.logger.setLevel(logging.DEBUG)
            handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
            handler.setFormatter(
                logging.Formatter("%(asctime)s | %(levelname)s | %(threadName)s | %(message)s")
            )
            self.logger.addHandler(handler)
        else:
            self.logger.setLevel(logging.CRITICAL + 10)

    def _init_llm_client(self):
        """
        Initialize the OpenAI-compatible client for the selected LLM backend.
        """
        api_key_env = "DEEPSEEK_API_KEY"
        base_url = "https://api.deepseek.com"

        self.client = OpenAI(
            api_key=os.environ.get(api_key_env),
            base_url=base_url
        )

    def call_llm(self, sys_prompt: str, message: str):
        """
        Call the LLM with a system prompt and user message.

        Parameters
        ----------
        sys_prompt : str
            System-level instruction for the LLM.
        message : str
            User message or task prompt.

        Returns
        -------
        tuple[str, datetime]
            The LLM response text and the localized timestamp
            at which the response was created.
        """
        kwargs = {
            "model": self.llm_model,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": message},
            ],
            "stream": False,
        }

        if self.thinking == "enabled":
            kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            if self.reasoning_effort is not None:
                kwargs["reasoning_effort"] = self.reasoning_effort
        else:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            kwargs["temperature"] = self.temperature

        response = self.client.chat.completions.create(**kwargs)

        date = datetime.fromtimestamp(response.created, tz=self.tz)
        response_text = response.choices[0].message.content

        return response_text, date

    def _generate_single_persona(self, idx, row, persona_system_prompt):
        """
        Generate a single persona description for one DataFrame row.

        Parameters
        ----------
        idx : int
            Index of the DataFrame row.
        row : pd.Series
            Row containing respondent attributes.
        persona_system_prompt : str
            System prompt instructing the LLM how to generate a persona.

        Returns
        -------
        tuple
            (row index, persona text or error message, generation datetime)
        """
        if pd.notna(row["llm_persona"]):
            return idx, None, None

        message = row["raw_persona"]
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
        """
        Generate personas for all rows in a DataFrame.

        Personas are stored in the columns:
        - 'llm_persona'
        - 'persona_generation_date'

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing Big5 attributes.
        persona_system_prompt : str
            System prompt defining how personas should be generated.

        Returns
        -------
        pd.DataFrame
            The updated DataFrame with persona information.
        persona_checkpoint_file.csv is automatically created and used in case of a crash.
        Personas.csv is the final .csv file with all the Persona data added to the original df.
        """
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

                if (i + 1) % self.save_every == 0:
                    df.to_csv(self.data_dir / "persona_checkpoint_file.csv", index=False)

        df.to_csv(self.data_dir / "Personas.csv", index=False)
        return df

            
    def _run_single_call(self, persona, idx, row, experiment, col_response):
        """
        Execute a single experimental LLM call for one row.

        Parameters
        ----------
        persona : str
            What persona condition should be used from [description, raw_persona, no_persona].
        idx : int
            Row index.
        row : pd.Series
            DataFrame row.
        experiment : Experiment
            Experiment definition containing prompt and instructions.
        col_response : str
            Column name where the response will be stored.

        Returns
        -------
        tuple
            (row index, response text or error message, response datetime)
        """
        # skip if already filled
        if pd.notna(row[col_response]):
            return idx, None, None
        match persona:
            case "description":
                system_prompt = textwrap.dedent(f"""
                    Your task is to behave in the way that this Person would behave:
                    {row['llm_persona']}
                    Respond in this format:
                    {experiment.sys_instruction}
                """).strip()
            case "raw_persona":
                system_prompt = textwrap.dedent(f"""
                    Your task is to behave in the way that this Person would behave:
                    {row['raw_persona']}
                    Respond in this format:
                    {experiment.sys_instruction}
                """).strip()
            case "no_persona":
                system_prompt = textwrap.dedent(f"""
                    Your task is to behave in the way that a Person would behave:
                    Respond in this format:
                    {experiment.sys_instruction}
                """).strip()
            case _:
                raise ValueError(f"Unknown persona mode: {persona}\nChoose one of: [description, raw_persona, no_persona]")
        self.logger.debug(f"Starting call | row={idx} | experiment={col_response} | persona={persona}")
        for attempt in range(1, self.max_retries + 1):
            try:
                response_text, date = self.call_llm(system_prompt, experiment.prompt)
                self.logger.info( # consider removing sysprompt and exp prompt logging
                    f"SUCCESS | row={idx} | experiment={col_response} | sys_prompt={system_prompt} | experiment_prompt={experiment.prompt}| attempt={attempt}"
                    )
                return idx, response_text, date 

            except Exception as e:
                msg = str(e).lower()
                retryable_errors = [
                    "timeout",
                    "timed out",
                    "429",
                    "rate limit",
                    "500",
                    "502",
                    "503",
                    "connection"
                ]
                if any(err in msg for err in retryable_errors):
                    if attempt == self.max_retries:
                        return idx, f"[ERROR: failed after {self.max_retries} retries: {str(e)}]", None
                    delay_time = self.base_delay * (2 ** (attempt - 1))
                    time.sleep(delay_time)
                    self.logger.warning(
                        f"TIMEOUT | row={idx} | experiment={col_response} |attempt={attempt}/{self.max_retries} | retrying in {delay_time:.1f}s"
                    )

                else:
                    self.logger.error(f"FAILED | row={idx} | experiment={col_response} | attempts={self.max_retries}")

                    return idx, f"[ERROR: {str(e)}]", None


    def run_experiment(
        self,
        df: pd.DataFrame,
        instructions: List[Dict[str, Experiment]],
        persona: Literal["description", "raw_persona", "no_persona"], 
        save_name: str
        ):
        """
        Run a set of experiments on all rows of a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame.
        instructions : list of dict
            List of experiment dictionaries mapping experiment names
            to `Experiment` objects.
        persona : str
            What persona condition should be used from [description, raw_persona, no_persona].
        save_name : str
            Base filename for the final CSV output.

        Returns
        -------
        pd.DataFrame
            The DataFrame augmented with experiment responses and timestamps.
        """
        for experiment_dict in tqdm(instructions, desc="Experiments"):
            for experiment_name, experiment in experiment_dict.items():

                col_response = f"{experiment_name}_experiment_{self.llm_model}"
                col_date = f"{col_response}_date"

                if col_response not in df.columns:
                    df[col_response] = None
                    
                if col_date not in df.columns:
                    df[col_date] = None

                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    futures = [
                        executor.submit(
                            self._run_single_call,
                            persona, row.name, row, experiment, col_response
                        )
                        for _, row in df.iterrows()
                    ]

                    for i, future in enumerate(tqdm(as_completed(futures), total=len(futures), leave=False)):
                        try:
                            idx, response_text, date = future.result()
                        except Exception as e:
                            self.logger.error(f"Unexpected error in future: {e}")
                            continue
                        
                        if response_text is None:
                            continue

                        df.at[idx, col_response] = response_text
                        df.at[idx, col_date] = date

                        if (i + 1) % self.save_every == 0:
                            df.to_csv(self.data_dir / f"experiment_checkpoint_{persona}.csv", index=False)

                df.to_csv(self.data_dir / f"experiment_checkpoint_{persona}.csv", index=False)

        df.to_csv(self.data_dir / f"{save_name}_{self.llm_model}_{persona}.csv", index=False)
        return df
