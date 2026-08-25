"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling an LLM.  Originally
written for the OpenAI API; this version has been patched to use the
``tgpt_backend`` package so the simulation runs *without* an OpenAI
API key, by routing every chat/completion call to the free ``tgpt``
CLI (or the ``pytgpt`` Python package) and using a deterministic
hash-based pseudo-embedding for memory retrieval.

The public function signatures are kept identical to the original
(``ChatGPT_single_request``, ``ChatGPT_request``, ``GPT4_request``,
``GPT_request``, ``ChatGPT_safe_generate_response``,
``GPT4_safe_generate_response``, ``safe_generate_response``,
``generate_prompt``, ``get_embedding``) so that no other file in the
generative_agents codebase needs to change.
"""
import json
import random
import time

# The original module did:
#     import openai
#     openai.api_key = openai_api_key
# We replace that with our tgpt-backed client.
from utils import *  # noqa: F401,F403  (kept for API compatibility)
from tgpt_backend import LLMClient, get_default_client


# Lazily build a process-wide client.  We accept the optional
# ``llm_client`` kwarg everywhere so tests can inject a stub.
_LLM_CLIENT: LLMClient = None  # type: ignore[assignment]


def _get_client() -> LLMClient:
    global _LLM_CLIENT
    if _LLM_CLIENT is None:
        _LLM_CLIENT = get_default_client()
    return _LLM_CLIENT


def temp_sleep(seconds=0.1):
  time.sleep(seconds)


def ChatGPT_single_request(prompt):
  temp_sleep()
  return _get_client().chat(prompt)


# ============================================================================
# #####################[SECTION 1: CHATGPT-3 STRUCTURE] ######################
# ============================================================================

def GPT4_request(prompt):
  """
  Given a prompt, make a request to the configured LLM and return the
  response.  We treat the ``gpt-4`` model the same as ``gpt-3.5-turbo`` —
  most free tgpt providers don't differentiate.  The ``model`` kwarg
  is forwarded to the backend when supported.
  """
  temp_sleep()
  try:
    return _get_client().chat(prompt, model="gpt-4")
  except Exception:
    print("LLM ERROR")
    return "LLM ERROR"


def ChatGPT_request(prompt):
  """
  Given a prompt, make a request to the configured LLM and return the
  response.
  """
  try:
    return _get_client().chat(prompt, model="gpt-3.5-turbo")
  except Exception:
    print("LLM ERROR")
    return "LLM ERROR"


def GPT4_safe_generate_response(prompt,
                                 example_output,
                                 special_instruction,
                                 repeat=3,
                                 fail_safe_response="error",
                                 func_validate=None,
                                 func_clean_up=None,
                                 verbose=False):
  prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose:
    print("LLM PROMPT")
    print(prompt)

  for i in range(repeat):
    try:
      curr_gpt_response = GPT4_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)

      if verbose:
        print("---- repeat count: \n", i, curr_gpt_response)
        print(curr_gpt_response)
        print("~~~~")

    except Exception:
      pass

  return False


def ChatGPT_safe_generate_response(prompt,
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
  # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "Example output json:\n"
  prompt += '{"output": "' + str(example_output) + '"}'

  if verbose:
    print("LLM PROMPT")
    print(prompt)

  for i in range(repeat):
    try:
      curr_gpt_response = ChatGPT_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]

      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)

      if verbose:
        print("---- repeat count: \n", i, curr_gpt_response)
        print(curr_gpt_response)
        print("~~~~")

    except Exception:
      pass

  return False


def ChatGPT_safe_generate_response_OLD(prompt,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
  if verbose:
    print("LLM PROMPT")
    print(prompt)

  for i in range(repeat):
    try:
      curr_gpt_response = ChatGPT_request(prompt).strip()
      if func_validate(curr_gpt_response, prompt=prompt):
        return func_clean_up(curr_gpt_response, prompt=prompt)
      if verbose:
        print(f"---- repeat count: {i}")
        print(curr_gpt_response)
        print("~~~~")

    except Exception:
      pass
  print("FAIL SAFE TRIGGERED")
  return fail_safe_response


# ============================================================================
# ###################[SECTION 2: ORIGINAL GPT-3 STRUCTURE] ###################
# ============================================================================

def GPT_request(prompt, gpt_parameter):
  """
  Given a prompt and a dictionary of GPT parameters, make a request and
  return the response.  ``gpt_parameter`` may include ``engine``,
  ``max_tokens``, ``temperature``, ``top_p``, ``frequency_penalty``,
  ``presence_penalty``, ``stream``, ``stop``.  We forward ``engine`` to
  the backend as a model hint; the rest are silently dropped because
  most free tgpt providers don't expose them.
  """
  temp_sleep()
  try:
    return _get_client().complete(prompt, **gpt_parameter)
  except Exception:
    print("TOKEN LIMIT EXCEEDED")
    return "TOKEN LIMIT EXCEEDED"


def generate_prompt(curr_input, prompt_lib_file):
  """
  Takes in the current input (e.g. comment that you want to classifiy) and
  the path to a prompt file. The prompt file contains the raw str prompt that
  will be used, which contains the following substr: !<INPUT>! -- this
  function replaces this substr with the actual curr_input to produce the
  final promopt that will be sent to the LLM server.
  ARGS:
    curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                INPUT, THIS CAN BE A LIST.)
    prompt_lib_file: the path to the promopt file.
  RETURNS:
    a str prompt that will be sent to the LLM.
  """
  if type(curr_input) == type("string"):
    curr_input = [curr_input]
  curr_input = [str(i) for i in curr_input]

  f = open(prompt_lib_file, "r")
  prompt = f.read()
  f.close()
  for count, i in enumerate(curr_input):
    prompt = prompt.replace(f"!<INPUT {count}>!", i)
  if "<commentblockmarker>###</commentblockmarker>" in prompt:
    prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
  return prompt.strip()


def safe_generate_response(prompt,
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False):
  if verbose:
    print(prompt)

  for i in range(repeat):
    curr_gpt_response = GPT_request(prompt, gpt_parameter)
    if func_validate(curr_gpt_response, prompt=prompt):
      return func_clean_up(curr_gpt_response, prompt=prompt)
    if verbose:
      print("---- repeat count: ", i, curr_gpt_response)
      print(curr_gpt_response)
      print("~~~~")
  return fail_safe_response


def get_embedding(text, model="text-embedding-ada-002"):
  """
  Return a numeric vector for ``text`` using the tgpt_backend embedding
  shim.  By default this is a deterministic, hash-based pseudo-embedding
  (1536 dims, L2-normalised).  See ``tgpt_backend.embeddings`` for how
  to plug in a real embedding model.
  """
  text = text.replace("\n", " ")
  if not text:
    text = "this is blank"
  return _get_client().embed(text)


if __name__ == '__main__':
  gpt_parameter = {"engine": "text-davinci-003", "max_tokens": 50,
                   "temperature": 0, "top_p": 1, "stream": False,
                   "frequency_penalty": 0, "presence_penalty": 0,
                   "stop": ['"']}
  curr_input = ["driving to a friend's house"]
  prompt_lib_file = "prompt_template/test_prompt_July5.txt"
  prompt = generate_prompt(curr_input, prompt_lib_file)

  def __func_validate(gpt_response):
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) > 1:
      return False
    return True
  def __func_clean_up(gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response

  output = safe_generate_response(prompt,
                                 gpt_parameter,
                                 5,
                                 "rest",
                                 __func_validate,
                                 __func_clean_up,
                                 True)

  print(output)
