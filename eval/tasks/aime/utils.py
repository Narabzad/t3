from collections import Counter
import os
import re
import signal
import json
from typing import Dict, List, Optional

import datasets

from lm_eval.utils import eval_logger

# Global variable to store retrieval data
RETRIEVAL_DATA = None
RETRIEVAL_TOP_K = 3  # Default top-k contexts to use
RETRIEVAL_OFFSET = 0  # Offset for mapping retrieval file lines to dataset indices

def load_retrieval_data(retrieval_file_path: str, top_k: int = 3, offset: int = 0):
    """Load retrieval data from JSONL file and create a lookup dictionary by index
    
    Args:
        retrieval_file_path: Path to JSONL file with retrieval results
        top_k: Number of top contexts to use
        offset: Offset to map retrieval file line numbers to dataset indices
                 (e.g., if retrieval file line 61 maps to dataset index 0, offset=60)
    """
    global RETRIEVAL_DATA, RETRIEVAL_TOP_K, RETRIEVAL_OFFSET
    RETRIEVAL_TOP_K = top_k
    RETRIEVAL_OFFSET = offset
    
    if retrieval_file_path is None or not os.path.exists(retrieval_file_path):
        eval_logger.warning(f"Retrieval file not found: {retrieval_file_path}. Running without retrieval.")
        RETRIEVAL_DATA = {}
        return
    
    RETRIEVAL_DATA = {}
    with open(retrieval_file_path, 'r') as f:
        for file_line_idx, line in enumerate(f):
            if line.strip():
                item = json.loads(line)
                ctxs = item.get('ctxs', [])
                # Take more contexts than top_k to account for empty ones
                # We'll filter to get exactly top_k non-empty contexts during processing
                # Take up to top_k * 2 contexts to ensure we have enough non-empty ones
                max_contexts_to_take = min(len(ctxs), top_k * 2) if ctxs else 0
                ctxs_top_k = ctxs[:max_contexts_to_take] if ctxs else []
                # Map retrieval file line to dataset index using offset
                # e.g., retrieval file line 61 (file_line_idx=60) -> dataset index 0 (60-60=0)
                dataset_idx = file_line_idx - offset
                if dataset_idx >= 0:  # Only store if offset results in valid index
                    RETRIEVAL_DATA[dataset_idx] = ctxs_top_k
    
    eval_logger.info(f"Loaded retrieval data for {len(RETRIEVAL_DATA)} items from {retrieval_file_path} (offset={offset})")

def doc_to_text_with_retrieval(doc: dict) -> str:
    """Format prompt with retrieval contexts using the examples-as-hints format"""
    question = doc.get("problem", doc.get("question"))
    
    # Get retrieval contexts from doc (stored during process_docs_with_retrieval)
    # Fallback to global RETRIEVAL_DATA if not in doc (for backwards compatibility)
    contexts = doc.get("_retrieval_contexts", [])
    
    if not contexts:
        # Fallback: try to get from global RETRIEVAL_DATA
        dataset_index = doc.get("_dataset_index")
        if RETRIEVAL_DATA is not None and dataset_index is not None and dataset_index in RETRIEVAL_DATA:
            ctxs = RETRIEVAL_DATA[dataset_index]
            seen_texts = set()  # Track seen texts to avoid duplicates
            for ctx in ctxs:
                retrieval_text = ctx.get('retrieval text', ctx.get('text', ''))
                if retrieval_text and retrieval_text.strip():  # Only add non-empty contexts
                    # Remove <think> tags from the retrieved text
                    retrieval_text = retrieval_text.replace('<think>', '').replace('</think>', '')
                    # Skip if we've already seen this exact text
                    if retrieval_text not in seen_texts:
                        seen_texts.add(retrieval_text)
                        contexts.append(retrieval_text)
                # Stop when we have top_k contexts
                if len(contexts) >= RETRIEVAL_TOP_K:
                    break
    
    # Build prompt with the new format
    if contexts:
        # Format: "Considering the following examples as hints..."
        prompt_parts = [
            "Considering the following examples as hints, try to answer the question.\n\nUse any useful hints and strategies from the retrieved examples.\n"
        ]
        
        # Add examples
        for i, ctx_text in enumerate(contexts, 1):
            prompt_parts.append(f"Example {i}: {ctx_text}")
        
        # Add main problem
        prompt_parts.append(f"\n\nmain problem:\n{question}")
        prompt_parts.append("Present the answer in LaTeX format: \\boxed{Your answer}")
        
        prompt = "\n".join(prompt_parts)
    else:
        # Fallback to standard format if no contexts
        prompt = QUERY_TEMPLATE.format(Question=question)
    
    return prompt

if os.getenv("PROMPTSTEP") is not None:
    QUERY_TEMPLATE = '{Question}\n\nThink for up to ' + os.getenv("PROMPTSTEP") + ' steps.'
elif os.getenv("PROMPTTOKEN") is not None:
    QUERY_TEMPLATE = '{Question}\n\nThink for up to ' + os.getenv("PROMPTTOKEN") + ' tokens.'
elif os.getenv("PROMPTLONG") is not None:
    QUERY_TEMPLATE = '{Question}\n\nAnswer after a long amount of thinking. If you feel like you are finished early, spend the extra time trying to double-check your work until you are absolutely sure that you have the correct answer.'
elif os.getenv("PROMPTSHORT") is not None:
    QUERY_TEMPLATE = '{Question}\n\nAnswer after a short amount of thinking. Do not spend excessive time double-checking your work.'
else:
    QUERY_TEMPLATE = '{Question}'

# The correct answer is an integer between $000$ and $999$, inclusive. Keep thinking until your answer is in the correct range.
# The correct answer is an integer between $000$ and $999$, inclusive.

print("QUERY_TEMPLATE: ", QUERY_TEMPLATE)

# Adapted from https://github.com/openai/simple-evals/blob/c0dba4c7bfbc17f786aec7bd7c3585a36ad81f23/common.py#L23
# (?i): Enables case-insensitive matching. This means "Answer", "answer", "ANSWER", etc., will all be matched.
# Answer: Matches the literal string "Answer" (case-insensitive due to (?i)).
# \s*: Matches zero or more whitespace characters (spaces, tabs, etc.) after "Answer". This accounts for cases where there might or might not be space between "Answer" and the colon (:).
# :: Matches the literal colon character :.
# \s*: Matches zero or more whitespace characters after the colon. This handles cases where there might be spaces between the colon and the actual answer.
# (.*): The .* matches zero or more of any character (including none), except for newlines unless re.DOTALL is used (which allows newlines to be matched too).
# Note: This does not match e.g. "**Final Answer:** A" as it only matches "Answer: A" or "Answer: A) 7" etc.
ANSWER_PATTERN = r"(?i)Answer\s*:\s*(.*)"

EXTRACTION_TEMPLATE_IDX = r"""
Look at the following attempt by a student and extract the student's answer. If it is equivalent (ignoring trivial simplifications) to any of the provided options, return the index of that option starting from 1. Else, return -1.

Examples:

    Options: ['2x+4', '2x', '4x']
    Attempt: The answer is 3+2x.

-1
(the student's answer is not among the options)

    Options: ['72,000']
    Attempt: 72000 \text{ cents}.

1
(always give benefit of the doubt to units and ignore formatting which makes the 1st option match)

    Options: ['2/(-3)', '2/3']
    Attempt: -1 * 2/3

1
(the 1st option matches after trivial simplifications which are fine)

    Options: ['x=5']
    Attempt: 5

1

    Options: ['\dfrac{33}{100}']
    Attempt: 0.33

1

    Options: ['75^\circ']
    Attempt: ...various calculations and explanations...hence the answer is $\boxed{x in 75}$.

1

    Options: ['(1,-3)', '(1,-1)', '(1,0)', '(1,-2)']
    Attempt: -2, 1

4
(ignore whitespace and other formatting which makes the 4th option match)

    Options: ['-2,1']
    Attempt: 1, -2

1
(likely a problem where multiple solutions are possible thus ignore order)

    Options: ['11', '100', '50', '-5', '12', '10']
    Attempt: ...$\boxed{12^{\mathrm{th}}}$.

5

    Options: ['2516_8']
    Attempt: 2516

1
(give benefit of the doubt for different bases)

    Options: ['11\sqrt2']
    Attempt: 11\sqrt{2}

1

    Options: ['11,\! 111,\! 111,\! 100']
    Attempt: 11111111100

1

    Options: ['\text{Navin}']
    Attempt: ...it is navin.

1

---

YOUR TASK


Respond with only the index of the matching option starting from 1 or -1 if there is absolutely no reasonable match. Do not include a rationale.

    Options: %(expression1)s
    Attempt: %(expression2)s
""".strip()


# https://github.com/openai/simple-evals/blob/580d359553a88584c11ce4efb97d49d9386e0d9e/common.py#L153C1-L156C45
def extract_answer_idx(sampler, options: List[str], attempt: str):
    prompt = EXTRACTION_TEMPLATE_IDX % {"expression1": options, "expression2": attempt}
    response = sampler([dict(content=prompt, role="user")])
    return response

import time
from typing import Any

import openai
from openai import OpenAI

class ChatCompletionSampler:
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content):
        return {"role": str(role), "content": content}

    def __call__(self, message_list) -> str:
        if self.system_message:
            message_list = [self._pack_message("system", self.system_message)] + message_list
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return ""
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception

def doc_to_text(doc: dict) -> str:
    return QUERY_TEMPLATE.format(Question=doc.get("problem", doc.get("question")))

def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process documents without retrieval - standard processing"""
    def _process_doc(doc: dict) -> dict:
        solution = doc.get("solution", doc.get("orig_solution", doc.get("orig_orig_solution")))
        problem = doc.get("problem", doc.get("question"))
        answer = doc.get("answer", doc.get("orig_answer", doc.get("orig_orig_answer")))
        doc_id = doc.get("id") or doc.get("index")
        
        if solution is None:
            print("Warning: No solution found; DOC:", doc)
        out_doc = {
            "problem": problem,
            "solution": solution,
            "answer": answer,
            "id": doc_id,
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc
    return dataset.map(_process_doc)

def normalize_text_for_matching(text: str) -> str:
    """Normalize text to handle symbol differences for question matching"""
    if not text:
        return ""
    import re
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove LaTeX dollar signs but keep content
    text = text.replace('$', '')
    # Normalize some common LaTeX differences
    text = text.replace('\\ldots', '...')
    text = text.replace('\\dots', '...')
    text = text.replace('\\triangle', 'triangle')
    text = text.replace('\\not=', '!=')
    text = text.replace('\\neq', '!=')
    # Lowercase for comparison
    return text.strip().lower()

def process_docs_aime23(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process AIME 2023 documents from qq8933/AIME_1983_2024 dataset"""
    # Filter for 2023 examples
    filtered_dataset = dataset.filter(lambda x: x.get("ID", "").startswith("2023-"))
    
    def _process_doc(doc: dict) -> dict:
        # Map fields from qq8933/AIME_1983_2024 format to expected format
        problem = doc.get("Question", doc.get("problem", doc.get("question")))
        answer = doc.get("Answer", doc.get("answer", doc.get("orig_answer")))
        doc_id = doc.get("ID", doc.get("id", doc.get("index")))
        solution = doc.get("solution", None)  # qq8933 dataset doesn't have solution field
        
        out_doc = {
            "problem": problem,
            "solution": solution,
            "answer": answer,
            "id": doc_id,
        }
        return out_doc
    
    return filtered_dataset.map(_process_doc)

def process_docs_aime23_with_retrieval(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process AIME 2023 documents with retrieval support"""
    retrieval_file_path = os.getenv("RETRIEVAL_FILE_PATH")
    top_k = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    
    # Set global RETRIEVAL_TOP_K
    global RETRIEVAL_TOP_K
    RETRIEVAL_TOP_K = top_k
    
    # Filter for 2023 examples
    filtered_dataset = dataset.filter(lambda x: x.get("ID", "").startswith("2023-"))
    
    # Build question-to-retrieval mapping if retrieval file is provided
    question_to_retrieval_idx = {}
    if retrieval_file_path and os.path.exists(retrieval_file_path):
        # Load retrieval entries for AIME 2023
        retrieval_entries = []
        with open(retrieval_file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if data.get('level') == 'AIME 2023':
                        retrieval_entries.append(data)
        
        # Create mapping by matching problem text
        aime23_list = list(filtered_dataset)
        for dataset_idx, aime_doc in enumerate(aime23_list):
            aime_problem = aime_doc.get("Question", aime_doc.get("problem", ""))
            aime_problem_norm = normalize_text_for_matching(aime_problem)
            
            best_match_idx = None
            best_score = 0
            
            for ret_idx, ret_entry in enumerate(retrieval_entries):
                ret_problem = ret_entry.get('problem', '')
                ret_problem_norm = normalize_text_for_matching(ret_problem)
                
                # Try exact match on first 100 chars
                if len(aime_problem_norm) > 50 and len(ret_problem_norm) > 50:
                    if aime_problem_norm[:100] == ret_problem_norm[:100]:
                        best_match_idx = ret_idx
                        break
                    # Or check word overlap
                    aime_words = set(aime_problem_norm.split()[:30])
                    ret_words = set(ret_problem_norm.split()[:30])
                    overlap = len(aime_words & ret_words)
                    if overlap > best_score and overlap > 10:
                        best_score = overlap
                        best_match_idx = ret_idx
            
            if best_match_idx is not None:
                question_to_retrieval_idx[dataset_idx] = best_match_idx
        
        eval_logger.info(f"Mapped {len(question_to_retrieval_idx)}/{len(aime23_list)} AIME 2023 questions to retrieval entries")
        
        # Store retrieval contexts in global RETRIEVAL_DATA using dataset indices
        global RETRIEVAL_DATA
        RETRIEVAL_DATA = {}
        for dataset_idx, ret_idx in question_to_retrieval_idx.items():
            if ret_idx < len(retrieval_entries):
                ctxs = retrieval_entries[ret_idx].get('ctxs', [])
                max_contexts = min(len(ctxs), top_k * 2) if ctxs else 0
                RETRIEVAL_DATA[dataset_idx] = ctxs[:max_contexts] if ctxs else []
    
    def _process_doc(doc: dict, idx: int) -> dict:
        # Map fields from qq8933/AIME_1983_2024 format to expected format
        problem = doc.get("Question", doc.get("problem", doc.get("question")))
        answer = doc.get("Answer", doc.get("answer", doc.get("orig_answer")))
        doc_id = doc.get("ID", doc.get("id", doc.get("index")))
        solution = doc.get("solution", None)
        
        # Get retrieval contexts for this doc index
        retrieval_contexts = []
        seen_texts = set()
        if RETRIEVAL_DATA is not None and idx in RETRIEVAL_DATA:
            ctxs = RETRIEVAL_DATA[idx]
            for ctx in ctxs:
                retrieval_text = ctx.get('retrieval text', ctx.get('text', ''))
                if retrieval_text and retrieval_text.strip():
                    # Remove <think> tags from the retrieved text
                    retrieval_text = retrieval_text.replace('<think>', '').replace('</think>', '')
                    # Skip if we've already seen this exact text
                    if retrieval_text not in seen_texts:
                        seen_texts.add(retrieval_text)
                        retrieval_contexts.append(retrieval_text)
                # Stop when we have top_k contexts
                if len(retrieval_contexts) >= RETRIEVAL_TOP_K:
                    break
        
        # Generate full prompt with retrieval contexts
        full_prompt = doc_to_text_with_retrieval({
            "problem": problem,
            "question": problem,
            "_retrieval_contexts": retrieval_contexts,
            "_dataset_index": idx
        })
        
        out_doc = {
            "problem": problem,
            "solution": solution,
            "answer": answer,
            "id": doc_id,
            "_dataset_index": idx,
            "_retrieval_contexts": retrieval_contexts,
            "_full_prompt": full_prompt,
        }
        return out_doc
    
    return filtered_dataset.map(_process_doc, with_indices=True)

def process_docs_aime22(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process AIME 2022 documents from qq8933/AIME_1983_2024 dataset"""
    # Filter for 2022 examples
    filtered_dataset = dataset.filter(lambda x: x.get("ID", "").startswith("2022-"))
    
    def _process_doc(doc: dict) -> dict:
        # Map fields from qq8933/AIME_1983_2024 format to expected format
        problem = doc.get("Question", doc.get("problem", doc.get("question")))
        answer = doc.get("Answer", doc.get("answer", doc.get("orig_answer")))
        doc_id = doc.get("ID", doc.get("id", doc.get("index")))
        solution = doc.get("solution", None)  # qq8933 dataset doesn't have solution field
        
        out_doc = {
            "problem": problem,
            "solution": solution,
            "answer": answer,
            "id": doc_id,
        }
        return out_doc
    
    return filtered_dataset.map(_process_doc)

def process_docs_aime22_with_retrieval(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process AIME 2022 documents with retrieval support"""
    retrieval_file_path = os.getenv("RETRIEVAL_FILE_PATH")
    top_k = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    
    # Set global RETRIEVAL_TOP_K
    global RETRIEVAL_TOP_K
    RETRIEVAL_TOP_K = top_k
    
    # Filter for 2022 examples
    filtered_dataset = dataset.filter(lambda x: x.get("ID", "").startswith("2022-"))
    
    # Build question-to-retrieval mapping if retrieval file is provided
    question_to_retrieval_idx = {}
    if retrieval_file_path and os.path.exists(retrieval_file_path):
        # Load retrieval entries for AIME 2022
        retrieval_entries = []
        with open(retrieval_file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if data.get('level') == 'AIME 2022':
                        retrieval_entries.append(data)
        
        # Create mapping by matching problem text
        aime22_list = list(filtered_dataset)
        for dataset_idx, aime_doc in enumerate(aime22_list):
            aime_problem = aime_doc.get("Question", aime_doc.get("problem", ""))
            aime_problem_norm = normalize_text_for_matching(aime_problem)
            
            best_match_idx = None
            best_score = 0
            
            for ret_idx, ret_entry in enumerate(retrieval_entries):
                ret_problem = ret_entry.get('problem', '')
                ret_problem_norm = normalize_text_for_matching(ret_problem)
                
                # Try exact match on first 100 chars
                if len(aime_problem_norm) > 50 and len(ret_problem_norm) > 50:
                    if aime_problem_norm[:100] == ret_problem_norm[:100]:
                        best_match_idx = ret_idx
                        break
                    # Or check word overlap
                    aime_words = set(aime_problem_norm.split()[:30])
                    ret_words = set(ret_problem_norm.split()[:30])
                    overlap = len(aime_words & ret_words)
                    if overlap > best_score and overlap > 10:
                        best_score = overlap
                        best_match_idx = ret_idx
            
            if best_match_idx is not None:
                question_to_retrieval_idx[dataset_idx] = best_match_idx
        
        eval_logger.info(f"Mapped {len(question_to_retrieval_idx)}/{len(aime22_list)} AIME 2022 questions to retrieval entries")
        
        # Store retrieval contexts in global RETRIEVAL_DATA using dataset indices
        global RETRIEVAL_DATA
        RETRIEVAL_DATA = {}
        for dataset_idx, ret_idx in question_to_retrieval_idx.items():
            if ret_idx < len(retrieval_entries):
                ctxs = retrieval_entries[ret_idx].get('ctxs', [])
                max_contexts = min(len(ctxs), top_k * 2) if ctxs else 0
                RETRIEVAL_DATA[dataset_idx] = ctxs[:max_contexts] if ctxs else []
    
    def _process_doc(doc: dict, idx: int) -> dict:
        # Map fields from qq8933/AIME_1983_2024 format to expected format
        problem = doc.get("Question", doc.get("problem", doc.get("question")))
        answer = doc.get("Answer", doc.get("answer", doc.get("orig_answer")))
        doc_id = doc.get("ID", doc.get("id", doc.get("index")))
        solution = doc.get("solution", None)
        
        # Get retrieval contexts for this doc index
        retrieval_contexts = []
        seen_texts = set()
        if RETRIEVAL_DATA is not None and idx in RETRIEVAL_DATA:
            ctxs = RETRIEVAL_DATA[idx]
            for ctx in ctxs:
                retrieval_text = ctx.get('retrieval text', ctx.get('text', ''))
                if retrieval_text and retrieval_text.strip():
                    # Remove <think> tags from the retrieved text
                    retrieval_text = retrieval_text.replace('<think>', '').replace('</think>', '')
                    # Skip if we've already seen this exact text
                    if retrieval_text not in seen_texts:
                        seen_texts.add(retrieval_text)
                        retrieval_contexts.append(retrieval_text)
                # Stop when we have top_k contexts
                if len(retrieval_contexts) >= RETRIEVAL_TOP_K:
                    break
        
        # Generate full prompt with retrieval contexts
        full_prompt = doc_to_text_with_retrieval({
            "problem": problem,
            "question": problem,
            "_retrieval_contexts": retrieval_contexts,
            "_dataset_index": idx
        })
        
        out_doc = {
            "problem": problem,
            "solution": solution,
            "answer": answer,
            "id": doc_id,
            "_dataset_index": idx,
            "_retrieval_contexts": retrieval_contexts,
            "_full_prompt": full_prompt,
        }
        return out_doc
    
    return filtered_dataset.map(_process_doc, with_indices=True)

def process_docs_aime25_with_retrieval(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process AIME 2025 documents with retrieval support"""
    retrieval_file_path = os.getenv("RETRIEVAL_FILE_PATH")
    top_k = int(os.getenv("RETRIEVAL_TOP_K", "3"))

    global RETRIEVAL_TOP_K
    RETRIEVAL_TOP_K = top_k

    # Build question-to-retrieval mapping if retrieval file is provided
    question_to_retrieval_idx = {}
    retrieval_entries = []
    if retrieval_file_path and os.path.exists(retrieval_file_path):
        # Load retrieval entries for AIME 2025 only
        with open(retrieval_file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if data.get('level') == 'AIME 2025':
                        retrieval_entries.append(data)

        # Create mapping by matching problem text
        dataset_list = list(dataset)
        for dataset_idx, aime_doc in enumerate(dataset_list):
            aime_problem = aime_doc.get("problem", aime_doc.get("question", aime_doc.get("Problem", "")))
            aime_problem_norm = normalize_text_for_matching(aime_problem)

            best_match_idx = None
            best_score = 0

            for ret_idx, ret_entry in enumerate(retrieval_entries):
                ret_problem = ret_entry.get('problem', '')
                ret_problem_norm = normalize_text_for_matching(ret_problem)

                if len(aime_problem_norm) > 50 and len(ret_problem_norm) > 50:
                    if aime_problem_norm[:100] == ret_problem_norm[:100]:
                        best_match_idx = ret_idx
                        break
                    aime_words = set(aime_problem_norm.split()[:30])
                    ret_words = set(ret_problem_norm.split()[:30])
                    overlap = len(aime_words & ret_words)
                    if overlap > best_score and overlap > 10:
                        best_score = overlap
                        best_match_idx = ret_idx

            if best_match_idx is not None:
                question_to_retrieval_idx[dataset_idx] = best_match_idx

        eval_logger.info(f"Mapped {len(question_to_retrieval_idx)}/{len(dataset_list)} AIME 2025 questions to retrieval entries")

        global RETRIEVAL_DATA
        RETRIEVAL_DATA = {}
        for dataset_idx, ret_idx in question_to_retrieval_idx.items():
            if ret_idx < len(retrieval_entries):
                ctxs = retrieval_entries[ret_idx].get('ctxs', [])
                max_contexts = min(len(ctxs), top_k * 2) if ctxs else 0
                RETRIEVAL_DATA[dataset_idx] = ctxs[:max_contexts] if ctxs else []

    def _process_doc(doc: dict, idx: int) -> dict:
        problem = doc.get("problem", doc.get("question", doc.get("Problem", "")))
        answer = doc.get("answer", doc.get("Answer", doc.get("orig_answer", "")))
        doc_id = doc.get("id", doc.get("ID", doc.get("index", idx)))
        solution = doc.get("solution", None)

        retrieval_contexts = []
        seen_texts = set()
        if RETRIEVAL_DATA is not None and idx in RETRIEVAL_DATA:
            ctxs = RETRIEVAL_DATA[idx]
            for ctx in ctxs:
                retrieval_text = ctx.get('retrieval text', ctx.get('text', ''))
                if retrieval_text and retrieval_text.strip():
                    retrieval_text = retrieval_text.replace('<think>', '').replace('</think>', '')
                    if retrieval_text not in seen_texts:
                        seen_texts.add(retrieval_text)
                        retrieval_contexts.append(retrieval_text)
                if len(retrieval_contexts) >= RETRIEVAL_TOP_K:
                    break

        full_prompt = doc_to_text_with_retrieval({
            "problem": problem,
            "question": problem,
            "_retrieval_contexts": retrieval_contexts,
            "_dataset_index": idx
        })

        out_doc = {
            "problem": problem,
            "solution": solution,
            "answer": answer,
            "id": doc_id,
            "_dataset_index": idx,
            "_retrieval_contexts": retrieval_contexts,
            "_full_prompt": full_prompt,
        }
        return out_doc

    return dataset.map(_process_doc, with_indices=True)


def process_docs_aime26_with_retrieval(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process AIME 2026 documents with retrieval support"""
    retrieval_file_path = os.getenv("RETRIEVAL_FILE_PATH")
    top_k = int(os.getenv("RETRIEVAL_TOP_K", "3"))

    global RETRIEVAL_TOP_K
    RETRIEVAL_TOP_K = top_k

    # Build question-to-retrieval mapping if retrieval file is provided
    question_to_retrieval_idx = {}
    retrieval_entries = []
    if retrieval_file_path and os.path.exists(retrieval_file_path):
        # Load retrieval entries for AIME 2026 only
        with open(retrieval_file_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if data.get('level') == 'AIME 2026':
                        retrieval_entries.append(data)

        # Create mapping by matching problem text
        dataset_list = list(dataset)
        for dataset_idx, aime_doc in enumerate(dataset_list):
            aime_problem = aime_doc.get("problem", aime_doc.get("question", aime_doc.get("Problem", "")))
            aime_problem_norm = normalize_text_for_matching(aime_problem)

            best_match_idx = None
            best_score = 0

            for ret_idx, ret_entry in enumerate(retrieval_entries):
                ret_problem = ret_entry.get('problem', '')
                ret_problem_norm = normalize_text_for_matching(ret_problem)

                if len(aime_problem_norm) > 50 and len(ret_problem_norm) > 50:
                    if aime_problem_norm[:100] == ret_problem_norm[:100]:
                        best_match_idx = ret_idx
                        break
                    aime_words = set(aime_problem_norm.split()[:30])
                    ret_words = set(ret_problem_norm.split()[:30])
                    overlap = len(aime_words & ret_words)
                    if overlap > best_score and overlap > 10:
                        best_score = overlap
                        best_match_idx = ret_idx

            if best_match_idx is not None:
                question_to_retrieval_idx[dataset_idx] = best_match_idx

        eval_logger.info(f"Mapped {len(question_to_retrieval_idx)}/{len(dataset_list)} AIME 2026 questions to retrieval entries")

        global RETRIEVAL_DATA
        RETRIEVAL_DATA = {}
        for dataset_idx, ret_idx in question_to_retrieval_idx.items():
            if ret_idx < len(retrieval_entries):
                ctxs = retrieval_entries[ret_idx].get('ctxs', [])
                max_contexts = min(len(ctxs), top_k * 2) if ctxs else 0
                RETRIEVAL_DATA[dataset_idx] = ctxs[:max_contexts] if ctxs else []

    def _process_doc(doc: dict, idx: int) -> dict:
        problem = doc.get("problem", doc.get("question", doc.get("Problem", "")))
        answer = doc.get("answer", doc.get("Answer", doc.get("orig_answer", "")))
        doc_id = doc.get("id", doc.get("ID", doc.get("index", idx)))
        solution = doc.get("solution", None)

        retrieval_contexts = []
        seen_texts = set()
        if RETRIEVAL_DATA is not None and idx in RETRIEVAL_DATA:
            ctxs = RETRIEVAL_DATA[idx]
            for ctx in ctxs:
                retrieval_text = ctx.get('retrieval text', ctx.get('text', ''))
                if retrieval_text and retrieval_text.strip():
                    retrieval_text = retrieval_text.replace('<think>', '').replace('</think>', '')
                    if retrieval_text not in seen_texts:
                        seen_texts.add(retrieval_text)
                        retrieval_contexts.append(retrieval_text)
                if len(retrieval_contexts) >= RETRIEVAL_TOP_K:
                    break

        full_prompt = doc_to_text_with_retrieval({
            "problem": problem,
            "question": problem,
            "_retrieval_contexts": retrieval_contexts,
            "_dataset_index": idx
        })

        out_doc = {
            "problem": problem,
            "solution": solution,
            "answer": answer,
            "id": doc_id,
            "_dataset_index": idx,
            "_retrieval_contexts": retrieval_contexts,
            "_full_prompt": full_prompt,
        }
        return out_doc

    return dataset.map(_process_doc, with_indices=True)


def process_docs_with_retrieval(dataset: datasets.Dataset) -> datasets.Dataset:
    """Process docs and load retrieval data if RETRIEVAL_FILE_PATH env var is set"""
    retrieval_file_path = os.getenv("RETRIEVAL_FILE_PATH")
    top_k = int(os.getenv("RETRIEVAL_TOP_K", "3"))
    offset = int(os.getenv("RETRIEVAL_OFFSET", "0"))
    
    # Load retrieval data if file is provided
    if retrieval_file_path:
        load_retrieval_data(retrieval_file_path, top_k, offset)
    
    def _process_doc(doc: dict, idx: int) -> dict:
        solution = doc.get("solution", doc.get("orig_solution", doc.get("orig_orig_solution")))
        problem = doc.get("problem", doc.get("question"))
        answer = doc.get("answer", doc.get("orig_answer", doc.get("orig_orig_answer")))
        doc_id = doc.get("id") or doc.get("index")
        
        # Get retrieval contexts for this doc index and store them in the doc
        retrieval_contexts = []
        seen_texts = set()  # Track seen texts to avoid duplicates
        if RETRIEVAL_DATA is not None and idx in RETRIEVAL_DATA:
            ctxs = RETRIEVAL_DATA[idx]
            for ctx in ctxs:
                retrieval_text = ctx.get('retrieval text', ctx.get('text', ''))
                if retrieval_text and retrieval_text.strip():  # Only add non-empty contexts
                    # Remove <think> tags from the retrieved text
                    retrieval_text = retrieval_text.replace('<think>', '').replace('</think>', '')
                    # Skip if we've already seen this exact text
                    if retrieval_text not in seen_texts:
                        seen_texts.add(retrieval_text)
                        retrieval_contexts.append(retrieval_text)
                # Stop when we have top_k contexts
                if len(retrieval_contexts) >= RETRIEVAL_TOP_K:
                    break
        
        # Generate the full prompt with retrieval contexts and store it
        # This ensures the exact prompt sent to the model is logged in the samples
        full_prompt = doc_to_text_with_retrieval({
            "problem": problem,
            "question": problem,
            "_retrieval_contexts": retrieval_contexts,
            "_dataset_index": idx
        })
        
        if solution is None:
            print("Warning: No solution found; DOC:", doc)
        out_doc = {
            "problem": problem,
            "solution": solution,
            "answer": answer,
            "id": doc_id,
            "_dataset_index": idx,  # Store index for retrieval matching
            "_retrieval_contexts": retrieval_contexts,  # Store retrieval contexts directly in doc
            "_full_prompt": full_prompt,  # Store the exact prompt that will be sent to the model
        }
        if getattr(doc, "few_shot", None) is not None:
            out_doc["few_shot"] = True
        return out_doc
    return dataset.map(_process_doc, with_indices=True)

def process_results(doc: dict, results: List[str]) -> Dict[str, int]:
    metrics = {"exact_match": None, "extracted_answers": []}
    # bp()
    # Multiple results -> we are measuring cov/maj etc
    if isinstance(results[0], list):
        results = results[0]
        n_res = len(results) # e.g. 64
        n_res_list = [2**i for i in range(1, int(n_res.bit_length()))] # e.g. [2, 4, 8, 16, 32, 64]
        metrics = {
            **metrics,
            "exact_matches": [],
            **{f"cov@{n}": -1 for n in n_res_list},
            **{f"maj@{n}": -1 for n in n_res_list},
        }

    if os.getenv("PROCESSOR", "") == "gpt-4o-mini":
        sampler = ChatCompletionSampler(model="gpt-4o-mini")
    else:
        print(f"Unknown processor: {os.getenv('PROCESSOR')}; set 'PROCESSOR=gpt-4o-mini' and 'OPENAI_API_KEY=YOUR_KEY' for best results.")
        sampler = None

    if isinstance(doc["answer"], str) and doc["answer"].isdigit():
        gt = str(int(doc["answer"])) # 023 -> 23
    else:
        gt = str(doc["answer"])
    split_tokens = ["<|im_start|>answer\n", "<|im_start|>"]

    for i, a in enumerate(results, start=1):
        if a is None:
            a = ""
        if split_tokens[0] in a:
            a = a.split(split_tokens[0])[-1]
        elif split_tokens[1] in a:
            a = a.split(split_tokens[1])[-1]
            if "\n" in a:
                a = "\n".join(a.split("\n")[1:])

        if (box := last_boxed_only_string(a)) is not None:
            a = remove_boxed(box)
        # re.DOTALL is key such that newlines are included e.g. if it does `Answer: Here is the solution:\n\n10`
        elif (matches := re.findall(ANSWER_PATTERN, a, re.DOTALL)) != []:
            a = matches[-1]  # Get the last match

        # AIME answers are from 000 to 999 so often it is a digit anyways
        if (a.isdigit()) and (gt.isdigit()):
            a = str(int(a)) # 023 -> 23
        elif sampler is not None:
            options = [gt] + list(set(metrics["extracted_answers"]) - {gt})
            if len(options) > 7:
                # Could switch back to exact returning like in AIME in that case
                # Problem with exact returning is that it sometimes messes up small things like a dollar sign
                print("Warning: Lots of options which may harm indexing performance:", options)            
            # This ensures that if doc['answer'] is \text{Evelyn} it is represented as such and not \\text{Evelyn}
            options_str = "[" + ", ".join(["'" + str(o) + "'" for o in options]) + "]"
            # a = extract_answer(sampler, options, a)
            idx = extract_answer_idx(sampler, options_str, a)
            if idx != "-1":
                if idx.isdigit():
                    idx = int(idx) - 1
                    if len(options) > idx >= 0:
                        a = options[idx]
                    else:
                        print("Warning: Index out of bounds; leaving answer unchanged\n", a, "\noptions", options_str, "\ndoc['answer']", gt, "\nidx", idx)
                else:
                    print("Warning: Processing did not produce integer index\na", a, "\noptions", options_str, "\ndoc['answer']", gt, "\nidx", idx)
        else:
            pass # TODO: Maybe add back legacy processing

        metrics["extracted_answers"].append(a)
        a = int(a == gt)
        if not(a): # Optional logging
            print("Marked incorrect\na " + metrics["extracted_answers"][-1] + "\ndoc['answer'] " + gt)
        if i == 1:
            metrics["exact_match"] = a
            if "exact_matches" in metrics:
                metrics["exact_matches"].append(a)
        elif i > 1:
            metrics["exact_matches"].append(a)
            if i in n_res_list:
                metrics[f"cov@{i}"] = int(1 in metrics["exact_matches"])
                metrics[f"maj@{i}"] = int(gt == Counter(metrics["extracted_answers"]).most_common(1)[0][0])

    return metrics

def last_boxed_only_string(string: str) -> Optional[str]:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval

def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        if s[: len(left)] != left:
            return s
        return s[len(left) :]

    left = "\\boxed{"

    if s[: len(left)] != left or s[-1] != "}":
        return s

    return s[len(left) : -1]


# ── HMMT tasks ────────────────────────────────────────────────────────────────

def _make_hmmt_process_docs(level_tag: str):
    """Factory: returns (process_docs_norag, process_docs_retrieval) for a given HMMT level."""

    def process_docs_norag(dataset):
        """Load HMMT problems from dataset (already loaded from local JSONL by lm_eval)."""
        def _proc(doc, idx):
            return {
                "problem": doc.get("problem", ""),
                "answer":  str(doc.get("answer", "")),
                "id":      doc.get("id", str(idx)),
                "solution": doc.get("solution", None),
                "_dataset_index": idx,
                "_retrieval_contexts": [],
                "_full_prompt": doc_to_text_with_retrieval({
                    "problem": doc.get("problem", ""),
                    "_retrieval_contexts": [],
                }),
            }
        return dataset.map(_proc, with_indices=True)

    def process_docs_retrieval(dataset):
        """Load HMMT problems with retrieval contexts from RETRIEVAL_FILE_PATH."""
        retrieval_file_path = os.getenv("RETRIEVAL_FILE_PATH")
        top_k = int(os.getenv("RETRIEVAL_TOP_K", "3"))
        global RETRIEVAL_TOP_K, RETRIEVAL_DATA
        RETRIEVAL_TOP_K = top_k

        # Build id→ctxs mapping from retrieval file
        id_to_ctxs = {}
        if retrieval_file_path and os.path.exists(retrieval_file_path):
            with open(retrieval_file_path) as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        id_to_ctxs[entry.get("id", "")] = entry.get("ctxs", [])
            eval_logger.info(f"Loaded {len(id_to_ctxs)} retrieval entries from {retrieval_file_path}")

        def _proc(doc, idx):
            doc_id = doc.get("id", str(idx))
            ctxs = id_to_ctxs.get(doc_id, [])
            retrieval_contexts = []
            seen = set()
            for ctx in ctxs:
                text = ctx.get("retrieval text", ctx.get("text", ""))
                text = text.replace("<think>", "").replace("</think>", "").strip()
                if text and text not in seen:
                    seen.add(text)
                    retrieval_contexts.append(text)
                if len(retrieval_contexts) >= top_k:
                    break
            return {
                "problem": doc.get("problem", ""),
                "answer":  str(doc.get("answer", "")),
                "id":      doc_id,
                "solution": doc.get("solution", None),
                "_dataset_index": idx,
                "_retrieval_contexts": retrieval_contexts,
                "_full_prompt": doc_to_text_with_retrieval({
                    "problem": doc.get("problem", ""),
                    "_retrieval_contexts": retrieval_contexts,
                }),
            }
        return dataset.map(_proc, with_indices=True)

    return process_docs_norag, process_docs_retrieval


# Register functions for each HMMT year
_hmmt_norag_feb2025,  _hmmt_ret_feb2025  = _make_hmmt_process_docs("HMMT Feb 2025")
_hmmt_norag_nov2025,  _hmmt_ret_nov2025  = _make_hmmt_process_docs("HMMT Nov 2025")
_hmmt_norag_feb2026,  _hmmt_ret_feb2026  = _make_hmmt_process_docs("HMMT Feb 2026")

process_docs_hmmt_feb_2025               = _hmmt_norag_feb2025
process_docs_hmmt_feb_2025_with_retrieval = _hmmt_ret_feb2025
process_docs_hmmt_nov_2025               = _hmmt_norag_nov2025
process_docs_hmmt_nov_2025_with_retrieval = _hmmt_ret_nov2025
process_docs_hmmt_feb_2026               = _hmmt_norag_feb2026
process_docs_hmmt_feb_2026_with_retrieval = _hmmt_ret_feb2026
