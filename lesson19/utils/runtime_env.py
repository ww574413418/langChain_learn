import os
from dotenv import load_dotenv
from utils.path_tool import get_project_root


def get_workspace_root() -> str:
    return os.path.dirname(get_project_root())


def get_env_path() -> str:
    return os.path.join(get_workspace_root(), "env")


def load_runtime_env() -> str:
    env_path = get_env_path()
    load_dotenv(env_path)
    return env_path


def setup_langsmith_defaults(project_name: str = "langchain_learn_lesson19") -> bool:
    load_runtime_env()

    api_key = os.getenv("LANGSMITH_API_KEY", "").strip()
    if not api_key:
        return False

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGSMITH_PROJECT", project_name)
    return True
