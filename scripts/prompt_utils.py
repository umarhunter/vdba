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

template = """You are a helpful assistant with access to neighborhood and demographic data. Respond in the following JSON format:

{format_instructions}

Question: {question}
Context: {context}

IMPORTANT INSTRUCTIONS:
1. First, determine if the question is:
   - A casual greeting/conversation (like "hello", "hi", "what can you do?") 
   - OR a data analysis question (asking about specific information from the context)

2. For CASUAL questions:
   - Respond naturally and briefly
   - Explain that you can help analyze neighborhood data, populations, demographics, etc.
   - Don't analyze the context unless specifically asked

3. For DATA ANALYSIS questions:
   - Carefully review ALL the context provided above - not just the first few items
   - For queries asking about "largest", "most", "highest", "smallest", etc., compare ALL items
   - Provide a thorough analysis in your thoughts
   - Give a clear, accurate answer based on the complete dataset

Your response MUST be a valid JSON object with exactly two fields:
- "thoughts": A string containing your reasoning process
- "answer": A string containing your final response

Example for casual question:
{{
    "thoughts": "This is a greeting, not a data query. I should respond friendly and explain my capabilities.",
    "answer": "Hello! I can help you analyze neighborhood data including populations, demographics, and locations. What would you like to know?"
}}

Example for data question:
{{
    "thoughts": "1. Examining all data points in the context...\n2. Comparing population values...\n3. Identifying the maximum...",
    "answer": "The neighborhood with the largest population is [name] with [number] residents."
}}
"""

# Create the prompt template with enforced formatting
QA_PROMPT = ChatPromptTemplate.from_template(
    template=template,
    partial_variables={"format_instructions": parser.get_format_instructions()}
)