"""Translate and beautify English from Chinese using LLM."""

import argparse
import logging
import os
import re

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
)


class EnvironmentVariableError(Exception):
    """Custom error for missing environment variables."""


def load_env() -> dict[str, str]:
    """Set environment variables."""
    if not load_dotenv():
        msg = (
            "No .env file found."
            "Please create one with the required environment variables."
        )
        raise EnvironmentVariableError(msg)

    env_vars = {var: os.getenv(var, "") for var in ["GLM_API", "API_URL"]}
    missing_vars = [var for var, value in env_vars.items() if value == ""]

    if missing_vars:
        msg = f"Missing environment variables: {', '.join(missing_vars)}"
        raise EnvironmentVariableError(msg)

    env_vars["model"] = "glm4"
    return env_vars


def check_language(s: str) -> str:
    """Check the language of the text."""
    if re.search("[\u4e00-\u9fff]", s):
        return "Chinese"
    if re.search("[a-zA-Z]", s):
        return "English"
    return "Unknown"


class TextTransformer:
    """Beautify English from Chinese."""

    def __init__(self) -> None:
        """Initialize the OpenAI client."""
        self.env_vars = load_env()
        self.model = self.env_vars["model"]
        self.client = OpenAI(
            api_key=self.env_vars["GLM_API"],
            base_url=self.env_vars["API_URL"],
        )

    def _process_text(self, prompt: dict[str, str]) -> str:
        """Process text based on the action."""
        system_message = ChatCompletionSystemMessageParam(
            role="system", content=prompt["system"]
        )
        user_message = ChatCompletionUserMessageParam(
            role="user", content=prompt["user"]
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                system_message,
                user_message,
            ],
        )

        if response.choices and response.choices[0].message.content:
            return response.choices[0].message.content
        return "No response."

    def _create_prompt(self, role: str, task: str, text: str, *, for_git: bool) -> dict:
        """Create a prompt dictionary."""
        if for_git:
            system_prompt = (
                f"You are an expert {role}, "
                "especially have experience in writing Github user friendly sentence."
            )
        else:
            system_prompt = f"You are an expert {role}"

        user_prompt = (
            f"{task.capitalize()} the following text, "
            "to improve clarity, conciseness, and coherence, "
            f"making them match the expression of native speakers.\n\n{text}"
        )
        return {"system": system_prompt, "user": user_prompt}

    def translate(self, text: str, *, for_git: bool) -> str:
        """Translate Chinese to English."""
        prompt = self._create_prompt(
            "translator",
            "please translate directly without explanation",
            text,
            for_git=for_git,
        )
        return self._process_text(prompt)

    def beautify(self, text: str, *, for_git: bool) -> str:
        """Beautify English text."""
        prompt = self._create_prompt(
            "writer",
            "please rewrite the text directly without explanation",
            text,
            for_git=for_git,
        )
        return self._process_text(prompt)


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    # TODO(luke396): More easier way to input text, support more lines
    parser = argparse.ArgumentParser(description="Translate and Beautify text.")
    parser.add_argument(
        "content", type=str, nargs="+", help="text to translate or beautify"
    )
    parser.add_argument(
        "-b", "--beautify", action="store_true", help="beautify English text"
    )
    parser.add_argument(
        "-t", "--translate", action="store_true", help="translate Chinese to English"
    )
    parser.add_argument(
        "-g", "--github", action="store_true", help="specify it used for Github message"
    )
    return parser.parse_args()


def handle_translation(
    trans: TextTransformer, text: str, *, for_git: bool = True
) -> None:
    """Handle translation."""
    if check_language(text) == "Chinese":
        eng = trans.translate(text, for_git=for_git)
        logging.info("%s", eng)
        logging.info("%s", trans.beautify(eng, for_git=for_git))
    else:
        logging.info(
            "Please provide Chinese text to translate."
            "If want to beauty English, use -b."
        )


def handle_beautification(
    trans: TextTransformer, text: str, *, for_git: bool = True
) -> None:
    """Handle beautification."""
    # TODO(luke396): Another output format for beautification
    if check_language(text) == "English":
        logging.info("%s", trans.beautify(text, for_git=for_git))
    else:
        logging.info(
            "Please provide English text to beautify."
            "If want to translate Chinese, use -t."
        )


def main() -> None:
    """Main function."""
    args = parse_arguments()
    text = " ".join(args.content)

    if not args.translate and not args.beautify:
        logging.info("Please specify an action. Use -h for help.")
        return

    trans = TextTransformer()
    for_git = args.github
    if args.translate:
        handle_translation(trans, text, for_git=for_git)
    if args.beautify:
        handle_beautification(trans, text, for_git=for_git)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
