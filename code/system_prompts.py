import textwrap
from typing import Literal

def get_system_prompt(persona: Literal["description", "raw_persona", "no_persona"]) -> str:
    if persona == "description":
        persona_information = textwrap.dedent("""
        Personality description::
        ------------------------------------------------------------
        {description}
        ------------------------------------------------------------
        """).strip()
    elif persona == "raw_persona":
        persona_information = textwrap.dedent("""
        ------------------------------------------------------------
        {raw_persona}
        ------------------------------------------------------------
        The personality scores range from 1 (very low) to 5 (very high) and were measured using the Big Five Aspect Scale (BFAS-40).
        The five broad personality traits and their corresponding subdomains are:
        - Extraversion (Enthusiasm + Assertiveness)
        - Agreeableness (Compassion + Politeness)
        - Conscientiousness (Industriousness + Orderliness)
        - Neuroticism (Volatility + Withdrawal)
        - Openness_Intellect (Openness + Intellect)
        The subdomain scores provide the most detailed personality information. The broad trait scores summarize these subdomains and are included for convenience.
        """).strip()

    elif persona == "no_persona":
        trait_instruction_1 = "Base your responses on the behavioral tendencies associated with the individual's age and gender when they are relevant."
        trait_instruction_2 = ""
        persona_information = textwrap.dedent(
        """
        ------------------------------------------------------------
        {no_persona}
        ------------------------------------------------------------
        """).strip()
    else:
        raise ValueError(f"Invalid personality profile type: {persona}")
    
    if persona in ["description", "raw_persona"]:
        trait_instruction_1 = "Base your responses on the behavioral tendencies associated with the provided personality profile while also considering the individual's age and gender when relevant."
        trait_instruction_2 = textwrap.dedent("""
        - Treat the provided personality profile as the primary source of behavioral information.
        - Do not exaggerate personality traits beyond what is implied by the provided scores.
        """).strip()

    SYSTEM_PROMPT = textwrap.dedent("""
    Your task is to simulate the responses of the following individual as accurately as possible.
    {persona_information}
    
    Use this information to infer how this individual would most likely think, judge, decide, and respond in each task.
    {trait_instruction_1}

    Guidelines:
    {trait_instruction_2}
    - Simulate the most probable response of the described individual, not the average human response.
    - When information is insufficient to determine a clear response, choose the option that is most consistent with the individual you are trying to simulate.
    - Respond exactly in the format requested by the user prompt.
    - Your objective is not to be correct or socially desirable, but to faithfully simulate the behavior of the described individual.
    """).strip()

    SYSTEM_PROMPT = SYSTEM_PROMPT.format(
        persona_information=persona_information,
        trait_instruction_1=trait_instruction_1,
        trait_instruction_2=trait_instruction_2,
    )
    return SYSTEM_PROMPT

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_csv("code/2_personas_test.csv")
    sys_prompt = get_system_prompt("no_persona")
    sys_prompt = sys_prompt.format(
        description=df["llm_persona"].iloc[0],
        raw_persona=df["raw_persona"].iloc[0],
        no_persona=f"Age: {df["Age"].iloc[0]}\nGender: {df["Gender"].iloc[0]}"
          )
    print(sys_prompt)
