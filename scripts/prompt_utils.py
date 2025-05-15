# scripts/prompt_utils.py
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# Define schemas for structured output
schemas = [
    ResponseSchema(
        name="thoughts",
        description="Detailed step-by-step reasoning process, including: "
                   "1. Initial analysis of the query "
                   "2. Key information identified "
                   "3. Relationships and patterns found "
                   "4. Reasoning steps taken "
                   "5. Confidence in conclusions"
    ),
    ResponseSchema(
        name="answer",
        description="Clear, concise final answer for the user, incorporating the key findings"
    )
]

parser = StructuredOutputParser.from_response_schemas(schemas)

# Define the prompt template with consistent variable names
template = """You are a helpful assistant analyzing documents.
For each question, follow these steps:

1. First, carefully think through the question
2. Analyze the relevant information from the context
3. Draw connections and identify patterns
4. Form logical conclusions
5. Express your confidence in the answer

{format_instructions}

Question: {question}
Context: {context}

Think through this step-by-step, showing your reasoning process before giving the final answer.
"""

# Create the prompt template
QA_PROMPT = ChatPromptTemplate.from_template(
    template=template,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)