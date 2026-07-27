from typing import Literal, Dict
import textwrap
from system_prompts import get_system_prompt
from experiment_runner import Experiment

def get_gamblers_fallacy_experiments(
        max_previous_outcomes: int,
        persona: Literal["description", "raw_persona", "no_persona"],
        group: int
        ) -> Dict[str, Experiment]:

    if group == 1:
        outcome_color = "red"
    else:
        outcome_color = "black"

    SYSTEM_PROMPT = get_system_prompt(persona=persona)
    BASE_PROMPT = textwrap.dedent("""                                                       
    The individual you are simulating is playing roulette and can either bet on black or red.
    Pick the choice that is most consistent with the described individual's behavioral tendencies.
    {history}What would the person choose? 
    Your response should be a single word containing the individuals choice.
    """).strip()

    experiments = {}

    for n in range(0, max_previous_outcomes + 1):
        if n == 0:
            history = ""
        elif n == 1:
            history = f"The last outcome was {outcome_color}. "
        else:
            history = f"The last {n} outcomes were {outcome_color}. "

        experiments[f"Gamblers_Fallacy_{n}"] = (
            Experiment(
                prompt=BASE_PROMPT.format(history=history),
                sys_instruction=SYSTEM_PROMPT
            
            )
        )
    return experiments


def get_loss_aversion_experiments(persona: Literal["description", "raw_persona", "no_persona"], group: int) -> Dict[str, Experiment]:
    
    def format_by_group(option_a: str, option_b: str, group: int) -> str:
        if group == 2:
            option_a, option_b = option_b, option_a
        return f"A: {option_a}\nB: {option_b}"
    
    system_prompt = get_system_prompt(persona=persona)

    base_prompt = (
        "You take part in a study evaluating peoples' choices among risky prospects. There is no 'correct' answer. "
        "Respond with either A or B.\n"
    )

    prompts = {
        "Loss_Aversion_1": format_by_group(
            "(2,500 with probability .33, 2,400 with probability .66, 0 with probability .01)",
            "(2,400 with certainty)",
            group
        ),
        "Loss_Aversion_2": format_by_group(
            "(2,500 with probability .33, 0 with probability .67)",
            "(2,400 with probability .34, 0 with probability .66)",
            group
        ),
        "Loss_Aversion_3": format_by_group(
            "(4,000 with probability .80)",
            "(3,000 with certainty)",
            group
        ),
        "Loss_Aversion_4": format_by_group(
            "(4,000 with probability .20)",
            "(3,000 with probability .25)",
            group
        ),

        "Loss_Aversion_5": format_by_group(
            "(50% chance to win a three-week tour of England, France, and Italy)",
            "(A one-week tour of England, with certainty)",
            group
        ),

        "Loss_Aversion_6": format_by_group(
            "(5% chance to win a three-week tour of England, France, and Italy)",
            "(10% chance to win a one-week tour of England)",
            group
        ),

        "Loss_Aversion_7": format_by_group(
            "(6,000 with probability .45)",
            "(3,000 with probability .90)",
            group
        ),

        "Loss_Aversion_8": format_by_group(
            "(6,000 with probability .001)",
            "(3,000 with probability .002)",
            group
        ),

        "Loss_Aversion_9": (
f"""Suppose you consider the possibility of insuring some property against damage, e.g., fire or theft. After examining the risks and the premium you find that you have no clear preference between the options of purchasing insurance or leaving the property uninsured.
It is then called to your attention that the insurance company offers a new program called probabilistic insurance. In this program you pay half of the regular premium. In case of damage, there is a 50 per cent chance that you pay the other half of the premium and the insurance company covers all the losses; and there is a 50 per cent chance that you get back your insurance payment and suffer all the losses.
For example, if an accident occurs on an odd day of the month, you pay the other half of the regular premium and your losses are covered; but if the accident occurs on an even day of the month, your insurance payment is refunded and your losses are not covered.
Recall that the premium for full coverage is such that you find this insurance barely worth its cost.
Under these circumstances, would you purchase probabilistic insurance:
{format_by_group("Yes", "No", group)}"""
        ),

        "Loss_Aversion_10": (
f"""Consider the following two-stage game. In the first stage, there is a probability of .75 to end the game without winning anything, and a probability of .25 to move into the second stage. If you reach the second stage you have a choice between
{format_by_group("(4,000 with probability .80)","(3,000 with certainty)", group)}
Your choice must be made before the game starts, i.e., before the outcome of the first stage is known."""
        ),

        "Loss_Aversion_11": (
f"""In addition to whatever you own, you have been given 1,000.
You are now asked to choose between
{format_by_group("(1,000 with probability .50)", "(500 with certainty)", group)}"""
        ),

        "Loss_Aversion_12": (
f"""In addition to whatever you own, you have been given 2,000.
You are now asked to choose between
{format_by_group("(1,000 with probability .50)", "(500 with certainty)", group)}"""
        ),

        "Loss_Aversion_13": format_by_group(
            "(6,000 with probability .25)",
            "(4,000 with probability .25; 2,000 with probability .25)",
            group
        ),

        "Loss_Aversion_13_inverse": format_by_group(
            "(-6,000 with probability .25)",
            "(-4,000 with probability .25; -2,000 with probability .25)",
            group
        ),

        "Loss_Aversion_14": format_by_group(
            "(5,000 with probability .001)",
            "(5 with certainty)",
            group
        ),

        "Loss_Aversion_14_inverse": format_by_group(
            "(-5,000 with probability .001)",
            "(-5 with certainty)",
            group
        ),
    }

    return {
        name: Experiment(
            prompt=base_prompt + prompt,
            sys_instruction=system_prompt
        )
        for name, prompt in prompts.items()
    }


def get_serial_position_effect_experiments(persona: Literal["description", "raw_persona", "no_persona"]) -> Dict[str, Experiment]:

    system_prompt = get_system_prompt(persona=persona)

    base_prompt = textwrap.dedent(
        """
        You are participating in a memory experiment.
        You will be shown a list of words, one word per line.
        Do not invent words.
        The order does not matter.
        Read the following list carefully. After reading it, recall as many words as you can remember.
        Output only the remembered words separated by commas.
        -----------------------------------------\n
        """)
    prompts = {
        "Serial_Position_Effect_20": "{Serial_Position_Effect_20}",
        "Serial_Position_Effect_40": "{Serial_Position_Effect_40}",
        "Serial_Position_Effect_80": "{Serial_Position_Effect_80}",
        }
    return {
        name: Experiment(
            prompt=base_prompt + prompt,
            sys_instruction=system_prompt
        )
        for name, prompt in prompts.items()
    }

def get_big_five_aspects_questionaire(persona: Literal["description", "raw_persona", "no_persona"]) -> Dict[str, Experiment]:

    system_prompt = get_system_prompt(persona=persona)
    # adapted from https://psytests.org/big5/bfasen-run.html?utm_source=chatgpt.com
    base_prompt = textwrap.dedent(
        """
        You will be provided with a phrase describing people's behaviors. 
        Describe yourself as you generally are now, not as you wish to be in the future.
        Describe yourself as you honestly see yourself, in relation to other people you know of the same sex as you are, and roughly your same age.
        Answer only with a single number from 1 to 5 that corresponds to how accurately the statement describes you.
        1 - Very Inaccurate
        2 - Moderately Inaccurate
        3 - Neither Accurate, Nor Inaccurate
        4 - Moderately Accurate
        5 - Very Accurate

        Rate the following statement:
        """
    )

    prompts = {
        "Volatility_1": (
            "get angry easily"
        ),
        "Volatility_2_R": (
            "am not easily annoyed"
        ),
        "Volatility_3": (
            "get easily agitated"
        ),
        "Volatility_4": (
            "can be stirred up easily"
        ),
        "Withdrawal_1_R": (
            "rarely feel depressed"
        ),
        "Withdrawal_2": (
            "worry about things"
        ),
        "Withdrawal_3_R": (
            "am not embarrassed easily"
        ),
        "Withdrawal_4": (
            "become overwhelmed by events"
        ),

        "Compassion_1": (
            "feel others' emotions"
        ),
        "Compassion_2": (
            "inquire about others' well-being"
        ),
        "Compassion_3_R": (
            "can't be bothered with other's needs"
        ),
        "Compassion_4": (
            "take an interest in other people's lives"
        ),
        "Politeness_1_R": (
            "insult people"
        ),
        "Politeness_2_R": (
            "believe that I am better than others"
        ),
        "Politeness_3": (
            "rarely put people under pressure"
        ),
        "Politeness_4_R": (
            "seek conflict"
        ),

        "Industriousness_1_R": (
            "waste my time"
        ),
        "Industriousness_2_R": (
            "find it difficult to get down to work"
        ),
        "Industriousness_3_R": (
            "mess things up"
        ),
        "Industriousness_4_R": (
            "am easily distracted"
        ),
        "Orderliness_1": (
            "like order"
        ),
        "Orderliness_2": (
            "keep things tidy"
        ),
        "Orderliness_3_R": (
            "am not bothered by disorder"
        ),
        "Orderliness_4": (
            "see that rules are observed"
        ),

        "Enthusiasm_1_R": (
            "keep others at a distance"
        ),
        "Enthusiasm_2_R": (
            "reveal little about myself"
        ),
        "Enthusiasm_3": (
            "warm up quickly to others"
        ),
        "Enthusiasm_4_R": (
            "am not a very enthusiastic person"
        ),
        "Assertiveness_1": (
            "take charge"
        ),
        "Assertiveness_2_R": (
            "lack the talent for influencing people"
        ),
        "Assertiveness_3_R": (
            "wait for others to lead the way"
        ),
        "Assertiveness_4": (
            "see myself as a good leader"
        ),

        "Openness_1": (
            "love to reflect on things"
        ),
        "Openness_2": (
            "see beauty in things that others might not notice"
        ),
        "Openness_3_R": (
            "seldom daydream"
        ),
        "Openness_4_R": (
            "seldom notice the emotional aspects of paintings and pictures"
        ),
        "Intellect_1": (
            "am quick to understand things"
        ),
        "Intellect_2_R": (
            "have difficulty understanding abstract ideas"
        ),
        "Intellect_3": (
            "can handle a lot of information"
        ),
        "Intellect_4_R": (
            "avoid difficult reading material"
        )
    }

    return {
        name: Experiment(
            prompt=base_prompt + prompt,
            sys_instruction=system_prompt
        )
        for name, prompt in prompts.items()
    }

if __name__ == "__main__":
    import pandas as pd
    import json

    df = pd.read_csv("/home/daniel/Desktop/Master_Thesis/code/2_personas_test.csv")

    test = get_serial_position_effect_experiments("description")

    # test one simulated person
    row = df.iloc[0]

    for name, experiment in test.items():

        map_size = name.split("_")[-1]  # extracts 20, 40, or 80

        word_map = json.loads(
            row[f"serial_position_effect_map_{map_size}"]
        )

        formatted_prompt = experiment.prompt.format(
            **{
                f"Serial_Position_Effect_{map_size}":
                "\n".join(word_map.keys())
            }
        )

        print(name)
        print("-" * 50)
        print(formatted_prompt)
        print()
        print("Position map:")
        print(word_map)
        print("=" * 80)