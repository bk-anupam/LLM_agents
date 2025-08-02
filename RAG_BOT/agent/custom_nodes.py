# /home/bk_anupam/code/LLM_agents/RAG_BOT/agent/custom_nodes.py
from typing import Any, Dict, List, cast
from langmem.short_term.summarization import (
    SummarizationNode,
    SummarizationResult,
    RunningSummary,
    PreprocessedMessages,
    _prepare_input_to_summarization_model,
    _preprocess_messages as original_preprocess_messages,
    asummarize_messages as original_asummarize,
)
from langchain_core.messages import BaseMessage
from langchain_core.prompts.chat import ChatPromptTemplate, ChatPromptValue
from RAG_BOT.logger import logger


def _logged_preprocess_messages(*args, **kwargs):
    """
    A wrapper around the original _preprocess_messages function that adds logging.
    """
    # Safe argument extraction for logging
    messages = args[0] if args else kwargs.get("messages")
    token_counter = kwargs.get("token_counter")
    max_tokens_before_summary = kwargs.get("max_tokens_before_summary")

    if messages and token_counter and max_tokens_before_summary is not None:
        total_tokens = token_counter(messages)
        logger.info(
            f"Summarization check: Total tokens in history = {total_tokens}, "
            f"Threshold = {max_tokens_before_summary}"
        )

        if total_tokens < max_tokens_before_summary:
            logger.info("Summarization condition NOT met. Passing all messages through.")
        else:
            logger.info("Summarization condition MET. Triggering summarization logic.")
    else:
        logger.warning("Could not perform summarization logging due to missing arguments.")

    # Define the arguments that the original function expects
    expected_args = {
        "messages": messages,
        "running_summary": kwargs.get("running_summary"),
        "max_tokens": kwargs.get("max_tokens"),
        "max_tokens_before_summary": kwargs.get("max_tokens_before_summary"),
        "max_summary_tokens": kwargs.get("max_summary_tokens"),
        "token_counter": kwargs.get("token_counter"),
    }
    # Call the original function with only the expected arguments
    return original_preprocess_messages(**expected_args)


def _custom_prepare_summarization_result(
    *,
    preprocessed_messages: PreprocessedMessages,
    messages: List[BaseMessage],
    existing_summary: RunningSummary | None,
    running_summary: RunningSummary | None,
    final_prompt: ChatPromptTemplate,
) -> SummarizationResult:
    """
    A corrected version of the original _prepare_summarization_result.
    It only applies the final_prompt if a new summary was actually generated in this step,
    preventing redundant summary messages from being added to the history.
    """
    # The key change: check if messages were actually summarized in this run.
    if preprocessed_messages.messages_to_summarize:
        # This block now only runs when a new summary has been created.
        # `running_summary` will be the newly generated summary object.
        total_summarized_messages = preprocessed_messages.total_summarized_messages + len(
            preprocessed_messages.messages_to_summarize
        )

        # This logic prevents re-adding a SystemMessage that might contain an old summary,
        # which is relevant when overwriting the main message list.
        include_system_message = preprocessed_messages.existing_system_message and not (
            existing_summary
            and existing_summary.summary
            in preprocessed_messages.existing_system_message.content
        )

        updated_messages = cast(
            ChatPromptValue,
            final_prompt.invoke(
                {
                    "system_message": [preprocessed_messages.existing_system_message]
                    if include_system_message
                    else [],
                    "summary": running_summary.summary,
                    "messages": messages[total_summarized_messages:],
                }
            ),
        )
        return SummarizationResult(
            running_summary=running_summary,
            messages=updated_messages.messages,
        )
    else:
        # No new summary was generated. Return the messages as they are,
        # only prepending the system message if it exists.
        # This prevents the final_prompt from being incorrectly applied.
        return SummarizationResult(
            running_summary=running_summary,  # Pass through the existing summary object for state continuity
            messages=(
                messages
                if preprocessed_messages.existing_system_message is None
                else [preprocessed_messages.existing_system_message] + messages
            ),
        )


async def logged_asummarize_messages(*args, **kwargs):
    """
    Wrapper for async summarization that uses the logged preprocessor.
    """
    try:
        # Safe argument extraction
        messages = args[0] if args else kwargs.get("messages")
        running_summary = kwargs.get("running_summary")
        model = kwargs.get("model")

        if messages is None or model is None:
            raise ValueError("`messages` and `model` are required for summarization.")

        preprocessed_messages = _logged_preprocess_messages(*args, **kwargs)

        if preprocessed_messages.existing_system_message:
            messages = messages[1:]

        if not messages:
            return SummarizationResult(
                running_summary=running_summary,
                messages=(
                    messages
                    if preprocessed_messages.existing_system_message is None
                    else [preprocessed_messages.existing_system_message] + messages
                ),
            )

        existing_summary = running_summary
        summarized_message_ids = (
            set(running_summary.summarized_message_ids) if running_summary else set()
        )
        if preprocessed_messages.messages_to_summarize:
            summary_messages = _prepare_input_to_summarization_model(
                preprocessed_messages=preprocessed_messages,
                running_summary=running_summary,
                existing_summary_prompt=kwargs.get("existing_summary_prompt"),
                initial_summary_prompt=kwargs.get("initial_summary_prompt"),
                token_counter=kwargs.get("token_counter"),
            )
            summary_response = await model.ainvoke(summary_messages)
            summarized_message_ids = summarized_message_ids | set(
                message.id for message in preprocessed_messages.messages_to_summarize
            )
            running_summary = RunningSummary(
                summary=summary_response.content,
                summarized_message_ids=summarized_message_ids,
                last_summarized_message_id=preprocessed_messages.messages_to_summarize[
                    -1
                ].id,
            )

        return _custom_prepare_summarization_result(
            preprocessed_messages=preprocessed_messages,
            messages=messages,
            existing_summary=existing_summary,
            running_summary=running_summary,
            final_prompt=kwargs.get("final_prompt"),
        )

    except Exception as e:
        logger.error(f"Error in logged_asummarize_messages: {e}", exc_info=True)
        # Fallback to original function in case of error
        return await original_asummarize(*args, **kwargs)


class LoggingSummarizationNode(SummarizationNode):
    """
    A custom summarization node that inherits from langmem's SummarizationNode
    and adds logging to indicate when summarization is actually performed.
    This helps in debugging and observing the agent's memory management behavior.
    """

    async def _afunc(self, input: dict[str, Any] | None = None, **kwargs: Any) -> Dict[str, Any]:
        """Override the async function to use our logged summarization logic."""
        messages, context = self._parse_input(input)

        summarization_result = await logged_asummarize_messages(
            messages,
            running_summary=context.get("running_summary"),
            model=self.model,
            max_tokens=self.max_tokens,
            max_tokens_before_summary=self.max_tokens_before_summary,
            max_summary_tokens=self.max_summary_tokens,
            token_counter=self.token_counter,
            initial_summary_prompt=self.initial_summary_prompt,
            existing_summary_prompt=self.existing_summary_prompt,
            final_prompt=self.final_prompt,
        )
        return self._prepare_state_update(context, summarization_result)

    def _func(self, input: dict[str, Any] | None = None, **kwargs: Any) -> Dict[str, Any]:
        raise NotImplementedError(
            "This custom node only supports asynchronous execution."
        )

    def _prepare_state_update(
        self, context: Dict[str, Any], summarization_result: SummarizationResult
    ) -> Dict[str, Any]:
        # Call the parent method to get the standard state update dictionary
        state_update = super()._prepare_state_update(context, summarization_result)

        if summarization_result.running_summary and (
            not context.get("running_summary")
            or summarization_result.running_summary.summary
            != context["running_summary"].summary
        ):
            summary_text = summarization_result.running_summary.summary
            num_summarized_ids = len(
                summarization_result.running_summary.summarized_message_ids
            )
            logger.info(
                f"Summarization performed. New summary created ({len(summary_text)} chars, summary preview: {summary_text[:200]}...). "
                f"Total messages summarized so far: {num_summarized_ids}."
            )
            logger.info(f"New summary snippet: {summary_text[:200]}...")
        else:
            logger.info("No new summary was generated in this step.")

        return state_update
