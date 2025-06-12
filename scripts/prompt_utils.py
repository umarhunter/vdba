from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Define schemas for structured output
schemas = [
    ResponseSchema(
        name="thoughts",
        description="Your detailed analysis process and reasoning steps"
    ),
    ResponseSchema(
        name="answer",
        description="Your final, concise answer based on the analysis"
    )
]

parser = StructuredOutputParser.from_response_schemas(schemas)

template = """You are a helpful assistant analyzing documents. Respond in the following JSON format:

{format_instructions}

Question: {question}
Context: {context}

Your response MUST be a valid JSON object with exactly two fields:
1. "thoughts": A string containing your step-by-step analysis
2. "answer": A string containing your final conclusion

Example response:
{{
    "thoughts": "1. Examining the agreement details...\n2. Checking the dates...\n3. Analyzing the terms...",
    "answer": "The clear, final answer based on the analysis."
}}
"""

# Create the prompt template with enforced formatting
QA_PROMPT = ChatPromptTemplate.from_template(
    template=template,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)