#!/usr/bin/env python
import sys
import warnings

from gemcrew.crew import Gemcrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    inputs = {
        'topic': '''
            Trudeau has been Prime Minister for 9 years. 

            In that time, former Immigration Minister Sean Fraser lost track of one million people who entered Canada and ran the immigration system "out of control," according to current Liberal Immigration Minister Marc Miller.

            All while both of them ignored warnings his immigration levels would drive up housing costs... and the Liberals also granted citizenship to an alleged ISIS terrorist.

            These Liberals can't fix what they broke.
        '''
    }
    crew_output = Gemcrew().crew().kickoff(inputs=inputs)

    if crew_output.json_dict:
        print(f"JSON Output: {crew_output.json_dict}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        Gemcrew().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while training the crew: {e}")

def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        Gemcrew().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")

def test():
    """
    Test the crew execution and returns the results.
    """
    inputs = {
        "topic": "AI LLMs"
    }
    try:
        Gemcrew().crew().test(n_iterations=int(sys.argv[1]), openai_model_name=sys.argv[2], inputs=inputs)

    except Exception as e:
        raise Exception(f"An error occurred while replaying the crew: {e}")
