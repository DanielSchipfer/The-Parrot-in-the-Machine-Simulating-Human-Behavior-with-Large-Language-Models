import textwrap
from typing import Literal

def get_system_prompt(persona: Literal["description", "raw_persona", "no_persona"]) -> str:
    if persona == "description":
         trait_instruction_1 = textwrap.dedent("""
        - a text containing the individuals personality profile 
        """).strip()

    elif persona == "raw_persona":
        trait_instruction_1 = textwrap.dedent("""
        - Big-Five personality trait scores (ranging from 1-5)
                - Extraversion (Contains Enthusiasm and Assertiveness) 
                - Agreeableness (Contains Compassion and Politeness)
                - Conscientiousness (Contains Industriousness and Orderliness)
                - Neuroticism (Contains Volatility and Withdrawal)
                - Openness_Intellect (Contains Intellect and Openness)
        """).strip()
        
    elif persona == "no_persona":
        trait_instruction_1 = ""
        trait_instruction_2 = "Base your responses on the behavioral tendencies associated with the individual's demographic characteristics when they are relevant."
        trait_instruction_3 = ""
    else:
        raise ValueError(f"Invalid personality profile type: {persona}")
    
    if persona in ["description", "raw_persona"]:
        trait_instruction_2 = "Base your responses on the behavioral tendencies associated with the provided personality profile while also considering the individual's demographic characteristics when they are relevant."
        trait_instruction_3 = (
        "- Treat the provided personality profile as the primary source of behavioral information.\n"
        "    - Do not exaggerate personality traits beyond what is implied by the provided scores.\n"
        )

    SYSTEM_PROMPT = textwrap.dedent(f"""
    You are participating in a scientific experiment on human behavior simulation.
    Your task is to simulate the responses of a single individual as accurately and consistently as possible.
    You will receive a description of one individual consisting of:
    - age
    - gender
    {trait_instruction_1}

    Use this information to infer how this individual would most likely think, judge, decide, and respond across given situations.
    {trait_instruction_2}

    Guidelines:
    - Simulate the most probable response of the described individual, not the average human response.
    - When information is insufficient to determine a clear response, choose the option that is most consistent with the individual you are trying to simulate.
    - Respond exactly in the format requested by the user prompt.
    {trait_instruction_3}
    Your objective is not to be correct or socially desirable, but to faithfully simulate the behavior of the described individual.
    """).strip()
    return SYSTEM_PROMPT