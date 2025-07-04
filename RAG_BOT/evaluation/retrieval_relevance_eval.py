from openevals.llm import create_llm_as_judge
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith.schemas import Run, Example 
from langsmith.evaluation import EvaluationResult
from RAG_BOT.config import Config

RAG_RETRIEVAL_RELEVANCE_PROMPT = """You are an expert data labeler evaluating retrieved context for relevance to the input. 
Your task is to assign a score based on the following rubric:

<Rubric>
- Relevant retrieved context:
  - Contain information that could help answer the input, even if incomplete.
  - May include superfluous information, but it should still be somewhat related to the input.

- Irrelevant retrieved context:
  - Contain no useful information for answering the input.
  - Are entirely unrelated to the input.
  - Contains misleading or incorrect information
  - Contains only tangentially related information with no practical utility
</Rubric>

Relevance Score:
A relevance score of True means that the FACTS contain ANY keywords or semantic meaning related to the QUESTION and are therefore relevant.
A relevance score of False means that the FACTS are completely unrelated to the QUESTION.

<Instruction>
- Read and understand the full meaning of the input (including edge cases)
- Formulate a list of facts and relevant context that would be needed to respond to the input
- Analyze the retrieved context to identify:
  - Information directly relevant to answering the query
  - Information partially relevant or contextually helpful
  - Information completely irrelevant to the query
- For each piece of information need identified in the previous step, determine:
  - Whether it is fully addressed by the retrieved documents (cite specific text)
  - Whether it is partially addressed (cite specific text)
  - Whether it is not addressed at all
- Note any facts needed to answer the input that are not found
- Note any context that are completely irrelevant, i.e. contain no relevant facts for answering the input
</Instruction>

<Reminder>  
- Focus solely on whether the retrieved context provides useful information for answering the input.
- Think deeply about why the context is or isn’t relevant.
- Use partial credit where applicable, recognizing context that is somewhat helpful even if incomplete.
</Reminder> 

<inputs>
{inputs}
</inputs>

<retrieved_context>
{context}
</retrieved_context>
"""

config = Config()

class RetrievalRelevanceGrade(BaseModel):
    score: bool = Field(description="True if the retrieved documents are relevant to the question, False otherwise.")
    reasoning: str = Field(description="Explanation for the hallucination score.")

# Create the evaluator instance once for efficiency.
_rag_ret_relevance_eval_instance = create_llm_as_judge(
    prompt=RAG_RETRIEVAL_RELEVANCE_PROMPT,
    judge=ChatGoogleGenerativeAI(model=config.JUDGE_LLM_MODEL_NAME, temperature=0),
    feedback_key="retrieval_relevance",
    output_schema=RetrievalRelevanceGrade,
)

def retrieval_relevance_evaluator(run: Run, example: Example):
    """
    Custom evaluator to measure hallucination using an LLM judge.
    It extracts the necessary components from the run and example objects.
    """
    # The `inputs` for the prompt is the question.
    input_dict = run.inputs.get("inputs") if run.inputs else {}
    # check if input_dict is not None and has 'question' key
    if input_dict and 'question' in input_dict:
        input_for_prompt = input_dict['question']
    else:
        input_for_prompt = None
            
    context_for_prompt = run.outputs.get("context") if run.outputs else None        
    
    eval_output = _rag_ret_relevance_eval_instance(
        inputs=input_for_prompt,                
        context=context_for_prompt
    )
    # eval_output is an instance of HallucinationEvaluation 
    # we need to convert it to EvaluationResult for compatibility with LangSmith
    return EvaluationResult(
        key="retrieval_relevance",
        score=eval_output.score,
        comment=eval_output.reasoning,
    )