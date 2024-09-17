"""
This script uses an LLM to convert formal set-theoretical definitions into natural language descriptions of sets. It then compares two sets to determine if they are equal or different. The LLM is configured to output structured responses following a predefined schema.
"""

import asyncio
from pydantic import BaseModel, Field
from typing import Literal
from neurologics.inference.message_models import LLMPromptContext, LLMConfig, StructuredTool, GeneratedJsonObject
from neurologics.inference.parallel_inference import ParallelAIUtilities

# First, we define the schema for the set
class LogicsSet(BaseModel):
    name: str
    formal_constraint: str = Field(default="a set theoretical definition of the set members using mathematical symbols")

class SetNaturalLanguage(LogicsSet):
    explanation: str = Field(default="a short explanation of the set in plain English")


# Now, let's try to convert a formal set to a natural language set
async def logic_to_en(name_first_set: str, name_second_set: str, formal_constraint_first_set: str, formal_constraint_second_set: str):
    llm_config = LLMConfig(client="openai", model="gpt-4o-mini", response_format="tool")
    structured_tool = StructuredTool(
        json_schema=SetNaturalLanguage.model_json_schema(),
        schema_name="SetNaturalLanguage",
        schema_description="High-level description of the elements belonging to a set",
        instruction_string="Please follow this JSON schema for your response:"
    )
    ai_utilities = ParallelAIUtilities()

    example_dog_set = LogicsSet(name=name_first_set, formal_constraint=formal_constraint_first_set)
    example_cat_set = LogicsSet(name=name_second_set, formal_constraint=formal_constraint_second_set)
    llm_prompt_context = LLMPromptContext(
        llm_config=llm_config,
        id="123",
        structured_output=structured_tool,
        new_message=f"Transform the following set {example_dog_set.model_dump()} defined formally into a set description in English"
    )

    dog_responses = await ai_utilities.run_parallel_ai_completion([llm_prompt_context])
    llm_prompt_context.new_message = f"Transform the following raw set {example_cat_set.model_dump()} defined in English into a formal set description"
    cat_responses = await ai_utilities.run_parallel_ai_completion([llm_prompt_context])
    dog_set = dog_responses[0].json_object
    cat_set = cat_responses[0].json_object
    print(dog_set)
    print(cat_set)

# Defining classes to compare two sets
class SetComparisonInput(BaseModel):
    name: str
    operator: Literal["==", "!="]
    first_input: LogicsSet
    second_input: LogicsSet

class SetComparisonOutput(BaseModel):
    response: bool
    reason: str

# Let's define a function to compare two sets
async def compare_sets(name_first_set: str, name_second_set: str, formal_constraint_first_set: str, formal_constraint_second_set: str):
    llm_config = LLMConfig(client="openai", model="gpt-4o-mini", response_format="tool")
    sets_to_compare = SetComparisonInput(name="", 
                                         operator="==", 
                                         first_input=LogicsSet(name=name_first_set, formal_constraint=formal_constraint_first_set), 
                                         second_input=SetNaturalLanguage(name=name_second_set, formal_constraint=formal_constraint_second_set)
                                        )
    
    structured_tool = StructuredTool(
        json_schema=SetComparisonOutput.model_json_schema(),
        schema_name="SetComparisonOutput",
        schema_description="Whether two sets are equal or not",
        instruction_string="Please follow this JSON schema for your response:"
    )

    llm_prompt_context = LLMPromptContext(
        llm_config=llm_config,
        id="123",
        structured_output=structured_tool,
        new_message=f"Are the two sets equal? {sets_to_compare.model_dump()}"
    )

    ai_utilities = ParallelAIUtilities()
    responses = await ai_utilities.run_parallel_ai_completion([llm_prompt_context])
    print(responses[0].json_object)
    
    
# Define the sets to translate andcompare
comparison_dict = {1: {"name_first_set": "Dogs group 1",
                       "name_second_set": "Dogs group 2",
                       "formal_constraint_first_set": "A = { x | x is an animal ∧ x barks ∧ x has four legs ∧ x has a wagging tail }",
                       "formal_constraint_second_set": "B = { x | x is an animal ∧ x barks ∧ x has three legs ∧ x has a wagging tail }"},
                   2: {"name_first_set": "Cats group 1",
                       "name_second_set": "Cats group 2",
                       "formal_constraint_first_set": "C = { x | x is an animal ∧ x meows ∧ x has four legs ∧ x has a wagging tail }",
                       "formal_constraint_second_set": "D = { x | x is an animal ∧ x meows ∧ x has three legs ∧ x has a wagging tail }"},
                   3: {"name_first_set": "Dogs group 1",
                       "name_second_set": "Cats group 1",
                       "formal_constraint_first_set": "E = { x | x is an animal ∧ x barks ∧ x has four legs ∧ x has a wagging tail }",
                       "formal_constraint_second_set": "F = { x | x is an animal ∧ x meows ∧ x has three legs ∧ x has a wagging tail }"},
                    4: {"name_first_set": "bla",
                       "name_second_set": "slkx",
                       "formal_constraint_first_set": "E = { x | x is an animal ∧ x barks }",
                       "formal_constraint_second_set": "F = { x | x is an animal ∧ x barks }"}
                    }

if __name__ == "__main__":
    for i in comparison_dict:
        name_first_set = comparison_dict[i]["name_first_set"]
        name_second_set = comparison_dict[i]["name_second_set"]
        formal_constraint_first_set = comparison_dict[i]["formal_constraint_first_set"]
        formal_constraint_second_set = comparison_dict[i]["formal_constraint_second_set"]

        print(f"--> Starting logic_to_en for {name_first_set} and {name_second_set}")
        asyncio.run(logic_to_en(name_first_set, name_second_set, formal_constraint_first_set, formal_constraint_second_set))
        print(f"\n --> Starting compare_sets for {name_first_set} and {name_second_set}")
        asyncio.run(compare_sets(name_first_set, name_second_set, formal_constraint_first_set, formal_constraint_second_set))
        print("\n")