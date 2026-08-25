import unittest
import tests.base as base
from os import getenv
from pytgpt.openai import OPENAI
from pytgpt.openai import AsyncOPENAI

API_KEY = getenv("OPENAI_API_KEY")


class TestOpenai(base.llmBase):
    def setUp(self):
        self.bot = OPENAI(API_KEY)
        self.prompt = base.prompt


class TestAsyncOpenai(base.AsyncProviderBase):

    def setUp(self):
        self.bot = AsyncOPENAI(API_KEY)
        self.prompt = base.prompt


if __name__ == "__main__":
    unittest.main()
