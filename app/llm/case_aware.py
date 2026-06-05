"""
app/llm/case_aware.py
─────────────────────
Specialized LLM prompting for case-constrained follow-up questions.

Key constraint: The LLM MUST refuse to answer questions outside the case scope.
"""
from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from app.config import Settings
from app.llm.groq_client import LLMClient
from app.models.session import CaseContextData

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# System prompt for case-constrained conversation
# ──────────────────────────────────────────────────────────────────────────────

_CASE_CONSTRAINED_SYSTEM_PROMPT = """You are a medical decision-support assistant specializing in oncology.

CRITICAL CONSTRAINT: You may ONLY answer questions directly related to the specific clinical case provided.

RULES:
1. If a question is relevant to the case → Answer thoroughly using the case context
2. If a question is outside the case scope → Politely refuse with a brief explanation
3. Do NOT provide general medical advice unrelated to this case
4. Do NOT answer hypothetical scenarios or different cases
5. Do NOT deviate from the scope of the current patient case

Example OUT-OF-SCOPE responses:
- "What is the general treatment for breast cancer?" → OUT OF SCOPE (not about THIS patient)
- "Should I use aspirin for pain?" → OUT OF SCOPE (not about THIS patient's condition)
- "Tell me about drug interactions" → OUT OF SCOPE (not specific to THIS case)

Example IN-SCOPE responses:
- "Why HER2 testing for this patient?" → IN SCOPE
- "What are the side effects of the recommended chemo?" → IN SCOPE
- "When should follow-up imaging be done?" → IN SCOPE

Format your response as:
- If relevant: Answer the question clearly, citing case details
- If not relevant: "I can only address questions specific to this case. Your question is outside the scope. Please ask about [patient details], [cancer type], or the treatment plan provided."
"""


class CaseAwareLLMClient:
    """
    Wraps LLMClient with case-constrained prompting for follow-up questions.
    """
    
    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client
    
    async def answer_follow_up(
        self,
        question: str,
        case_context: CaseContextData,
    ) -> AsyncIterator[str]:
        """
        Answer a follow-up question constrained to the case scope.
        
        Args:
            question: Doctor's follow-up question
            case_context: Original case data and initial response
        
        Yields:
            Response tokens, constrained to case scope
        """
        
        # Build rich context for the LLM
        case_summary = self._build_case_summary(case_context)
        
        messages = [
            {
                "role": "user",
                "content": (
                    f"CASE CONTEXT:\n{case_summary}\n\n"
                    f"INITIAL DECISION-SUPPORT OUTPUT:\n{case_context.initial_llm_response}\n\n"
                    f"DOCTOR'S FOLLOW-UP QUESTION:\n{question}"
                ),
            }
        ]
        
        logger.info(
            f"Answering follow-up for {case_context.cancer_type} case"
        )
        
        # Stream response using the case-constrained system prompt
        async for token in self._llm.stream_answer(
            messages=messages,
            system_prompt=_CASE_CONSTRAINED_SYSTEM_PROMPT,
        ):
            yield token
    
    def _build_case_summary(self, case_context: CaseContextData) -> str:
        """
        Build a concise summary of the case for context.
        
        Args:
            case_context: Case data
        
        Returns:
            Formatted case summary
        """
        
        summary_lines = [
            f"Cancer Type: {case_context.cancer_type.upper()}",
            f"Tool Used: {case_context.tool_name}",
            f"Parameters Submitted: {case_context.parameters_sent}",
            "",
            "Patient Data:",
        ]
        
        # Format patient data nicely
        for key, value in case_context.patient_data.items():
            # Convert snake_case to Title Case
            display_key = key.replace('_', ' ').title()
            summary_lines.append(f"  • {display_key}: {value}")
        
        if case_context.request_context:
            summary_lines.append("")
            summary_lines.append(f"Initial Request: {case_context.request_context}")
        
        return "\n".join(summary_lines)