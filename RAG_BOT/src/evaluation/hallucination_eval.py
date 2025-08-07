from openevals.prompts import HALLUCINATION_PROMPT
from openevals.llm import create_llm_as_judge
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith.schemas import Run, Example 
from langsmith.evaluation import EvaluationResult
from RAG_BOT.src.config.config import Config

config = Config()

class HallucinationEvaluation(BaseModel):
    score: float = Field(description="A score indicating the degree of hallucination.")
    reasoning: str = Field(description="Explanation for the hallucination score.")

# Create the evaluator instance once for efficiency.
_hallucination_evaluator_instance = create_llm_as_judge(
    prompt=HALLUCINATION_PROMPT,
    judge=ChatGoogleGenerativeAI(model=config.JUDGE_LLM_MODEL_NAME, temperature=0),
    feedback_key="hallucination",
    output_schema=HallucinationEvaluation,
)

def hallucination_evaluator(run: Run, example: Example):
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
        
    output_for_prompt = run.outputs.get("answer") if run.outputs else None    
    context_for_prompt = run.outputs.get("context") if run.outputs else None    
    reference_for_prompt = example.outputs.get("answer") if example and example.outputs else None    
    
    eval_output = _hallucination_evaluator_instance(
        inputs=input_for_prompt,
        outputs=output_for_prompt,
        reference_outputs=reference_for_prompt,
        context=context_for_prompt
    )
    # eval_output is an instance of HallucinationEvaluation 
    # we need to convert it to EvaluationResult for compatibility with LangSmith
    return EvaluationResult(
        key="hallucination",
        score=eval_output.score,
        comment=eval_output.reasoning,
    )