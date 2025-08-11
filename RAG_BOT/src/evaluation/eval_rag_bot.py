import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI
from langsmith import Client, aevaluate
from langsmith.evaluation import LangChainStringEvaluator
from langsmith.schemas import Run, Example 
from langdetect import detect
from RAG_BOT.src.agent.graph_builder import build_agent
from RAG_BOT.src.config.config import Config
from RAG_BOT.src.persistence.vector_store import VectorStore
from RAG_BOT.src.logger import logger
from RAG_BOT.src.agent.state import AgentState
from langchain_core.messages import HumanMessage, AIMessage
from RAG_BOT.src.evaluation.hallucination_eval import hallucination_evaluator
from RAG_BOT.src.evaluation.retrieval_relevance_eval import retrieval_relevance_evaluator
from RAG_BOT.src.json_parser import JsonParser


ls_client = Client()
dataset_name = "eval_rag_bot_dataset"
config = Config()
json_parser = JsonParser()


async def get_rag_agent(config: Config = None):        
    persist_directory = config.VECTOR_STORE_PATH    
    vector_store_instance = VectorStore(persist_directory=persist_directory, config=config)
    vectordb = vector_store_instance.get_vectordb()
    logger.info("VectorStore initialized.")      
    agent = await build_agent(vectordb=vectordb, config_instance=config)
    logger.info("RAG agent initialized.")            
    return agent


def prepare_langchain_eval_data(run: Run, example: Example):
    """Prepares data for the LangChainStringEvaluator to specify which keys to use."""
    return {
        "prediction": run.outputs.get("answer"),
        "reference": example.outputs.get("answer"),
        "input": example.inputs.get("question"),
    }


def create_eval_fn(agent):
    """Factory function that creates an eval_fn with agent in closure"""
    async def eval_fn(inputs: dict) -> dict:
        user_question = inputs['question']
        language_code = detect(user_question) 
        final_answer = None
        
        # Initialize state correctly
        initial_state = AgentState(
            messages=[HumanMessage(content=user_question)],
            original_query=None,
            current_query=None,
            context=None,
            retry_attempted=False,
            evaluation_result=None,
            language_code=language_code,
            documents=None,
            web_search_attempted=False,
            last_retrieval_source=None
        )         
        
        final_response = await agent.ainvoke(initial_state, {"recursion_limit": 15})
        final_answer_message = final_response['messages'][-1]
        
        if isinstance(final_answer_message, AIMessage):                
            parsed_json = json_parser.parse_json_answer(final_answer_message.content)
            if parsed_json and "answer" in parsed_json:
                final_answer = parsed_json.get("answer")
            else:
                final_answer = "Final answer content is not valid JSON."
        
        # Return both the answer and the context used to generate it.
        final_context = final_response.get("context", "")
        return {"answer": final_answer, "context": final_context}
    
    return eval_fn


async def evaluate_rag_agent(config: Config):
    agent = await get_rag_agent(config)
    # Create eval_fn with agent in closure       
    eval_fn = create_eval_fn(agent)  
    # create prebuilt evaluator instance
    eval_llm = ChatGoogleGenerativeAI(model=config.LLM_MODEL_NAME, temperature=0)
    cot_qa_evaluator = LangChainStringEvaluator(
        "cot_qa", 
        config={"llm": eval_llm},
        prepare_data=prepare_langchain_eval_data
    )
    results = await aevaluate(
        eval_fn,
        data=dataset_name,
        evaluators=[cot_qa_evaluator, hallucination_evaluator, retrieval_relevance_evaluator],
        client=ls_client
    )
    return results


if __name__ == "__main__":
    async def run_evaluation():
        config = Config()        
        results = await evaluate_rag_agent(config)
        print(f"Evaluation results: {results}")

    asyncio.run(run_evaluation())
