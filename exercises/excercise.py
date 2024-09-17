import asyncio
from pydantic import BaseModel, Field
from typing import Literal
from neurologics.inference.message_models import LLMPromptContext, LLMConfig, StructuredTool, GeneratedJsonObject
from neurologics.inference.parallel_inference import ParallelAIUtilities

class RawSet(BaseModel):
    name: str
    eng_constraint: str = Field(default="a description of the set in natural language")

class SetBuilder(RawSet):
    formal_constraint: str = Field(default="a set theoretical definition of the set members using mathematical symbols")
    explanation: str = Field(default="a short explanation of the set for mathematical beginner")

class SetInputComparison(BaseModel):
    name: str
    operator: Literal["==", "!="]
    first_input: SetBuilder
    second_input: SetBuilder

class SetOutputComparison(BaseModel):
    response: bool
    reason: str

async def main():
    llm_config = LLMConfig(client="openai", model="gpt-4o-mini", response_format="tool")
    structured_tool = StructuredTool(
        json_schema=SetBuilder.model_json_schema(),
        schema_name="SetBuilder",
        schema_description="A set of elements that can be used to create a set",
        instruction_string="Please follow this JSON schema for your response:"
    )
    ai_utilities = ParallelAIUtilities()

    example_dog_set = RawSet(name="Dogs", eng_constraint="Dogs are animals that bark with four legs and can have a wagging tail")
    example_cat_set = RawSet(name="Cats", eng_constraint="Cats are animals that meow with four legs and can have a wagging tail")
    llm_prompt_context = LLMPromptContext(
        llm_config=llm_config,
        id="123",
        structured_output=structured_tool,
        new_message=f"Transform the following raw set {example_dog_set.model_dump()} defined in English into a formal set description"
    )


    dog_responses = await ai_utilities.run_parallel_ai_completion([llm_prompt_context])
    llm_prompt_context.new_message = f"Transform the following raw set {example_cat_set.model_dump()} defined in English into a formal set description"
    cat_responses = await ai_utilities.run_parallel_ai_completion([llm_prompt_context])
    dog_set = dog_responses[0].json_object
    cat_set = cat_responses[0].json_object
    print(dog_set)
    print(cat_set)
    assert isinstance(dog_set, GeneratedJsonObject)
    assert isinstance(cat_set, GeneratedJsonObject)

    set_input_comparison = SetInputComparison(
        name="Dog vs Cat",
        operator="!=",
        first_input=SetBuilder.model_validate(dog_set.object),
        second_input=SetBuilder.model_validate(cat_set.object)
    )

    comparison_tool = StructuredTool(
        json_schema=SetOutputComparison.model_json_schema(),
        schema_name="SetOutputComparison",
        schema_description="A comparison of two sets",
        instruction_string="Please follow this JSON schema for your response:"
    )
    llm_prompt_context = LLMPromptContext(
        llm_config=llm_config,
        id="123",
        structured_output=comparison_tool,
        new_message=f"Is the first set equal to the second set? {set_input_comparison.model_dump()}"
    )
    responses = await ai_utilities.run_parallel_ai_completion([llm_prompt_context])
    print(responses[0].json_object)

if __name__ == "__main__":
    asyncio.run(main())
