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
    Your response should be a single word containing the individuals choice.
    {history}What would the person choose?
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
        "You can answer with either A or B.\n"
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
{format_by_group("(1,000 with probability .50)", "(500 with certainty", group)}"""
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
            "5 with certainty)",
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
        """You are participating in a memory experiment.
        You will be presented with a list of words, one at a time.
        After the list is presented, you will be asked to recall as many words as you can remember from the list.\n"""
    )

    prompts = {
        "Primacy_Effect": (
            "INSERT PROMPT FOR PRIMACY EFFECT HERE"
        ),
        "Recency_Effect": (
            "INSERT PROMPT FOR RECENCY EFFECT HERE"
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
    # gamblers_fallacy_experiments = get_gamblers_fallacy_experiments(max_previous_outcomes=10, persona="raw_persona")

    # # Print the names of the experiments
    # print("Gamblers Fallacy prompt:",(gamblers_fallacy_experiments["Gamblers_Fallacy_0"].prompt))
    # print("-"*50)
    # print("Gamblers Fallacy sys:",(gamblers_fallacy_experiments["Gamblers_Fallacy_0"].sys_instruction))
    # loss_aversion = get_loss_aversion_experiments("description", 1)
    # print(loss_aversion["Loss_Aversion_1"].sys_instruction)
    # print("---------------------------------------------------------------")
    # print(loss_aversion["Loss_Aversion_1"].prompt)
    # for name, experiment in loss_aversion.items():
    #     print(f"{name}")
    #     print("-" * 50)
    #     print(experiment.prompt)
    #     print()

    gamblers_fallacy = get_gamblers_fallacy_experiments(10, "description", 2)
    for name, experiment in gamblers_fallacy.items():
        print(f"{name}")
        print("-" * 50)
        print(experiment.prompt)
        print()
