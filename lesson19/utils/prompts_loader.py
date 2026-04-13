from .config_handler import prompts_config
from .path_tool import get_abs_path
from .logger_handler import logger as log

def load_system_prompt()->str:
    '''
    load prompt
    :return:
    '''
    try:
        system_prompt_path = get_abs_path(prompts_config["main_prompt_path"])
    except KeyError as e:
        log.error(f"not found main_prompt_path please check your config:{prompts_config}")
        raise e

    try:
        with open(system_prompt_path,"r",encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as e:
        log.error(f"not found system prompt file:{system_prompt_path}")
        raise e

def load_rag_prompt()-> str:
    '''
    load prompt
    :return:
    '''
    try:
        rag_prompt_path = get_abs_path(prompts_config["rag_summarize_prompt_path"])
    except KeyError as e:
        log.error(f"not found rag_prompt_path please check your config:{prompts_config}")
        raise e

    try:
        with open(rag_prompt_path,"r",encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as e:
        log.error(f"not found system prompt file:{rag_prompt_path}")
        raise e

def load_report_prompt()->str:
    '''
    load prompt
    :return:
    '''
    try:
        report_prompt_path = get_abs_path(prompts_config["report_prompt_path"])
    except KeyError as e:
        log.error(f"not found report_prompt_path please check your config:{prompts_config}")
        raise e

    try:
        with open(report_prompt_path,"r",encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as e:
        log.error(f"not found system prompt file:{report_prompt_path}")
        raise e


def load_refine_query_prompt()->str:
    '''
    load prompt
    :return:
    '''
    try:
        refine_query_prompt_path = get_abs_path(prompts_config["refine_query_prompt_path"])
    except KeyError as e:
        log.error(f"not found report_prompt_path please check your config:{prompts_config}")
        raise e

    try:
        with open(refine_query_prompt_path,"r",encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as e:
        log.error(f"not found system prompt file:{refine_query_prompt_path}")
        raise e

def extract_userprofile_prompt() -> dict:
    '''
    从query中提取用户profile的patch
    '''
    try:
        extract_user_query_prompt_path = get_abs_path(prompts_config["extract_user_query_prompt_path"])
    except KeyError as e:
        log.error(f"not found report_prompt_path please check your config:{prompts_config}")
        raise e

    try:
        with open(extract_user_query_prompt_path,"r",encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as e:
        log.error(f"not found system prompt file:{extract_user_query_prompt_path}")
        raise e

def select_memory_plan_prompt() -> dict:
    '''
    导入记忆选择prompt
    '''
    try:
        select_memoryplan_prompt_path = get_abs_path(prompts_config["select_memoryplan_prompt_path"])
    except KeyError as e:
        log.error(f"not found report_prompt_path please check your config:{prompts_config}")
        raise e

    try:
        with open(select_memoryplan_prompt_path,"r",encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as e:
        log.error(f"not found system prompt file:{select_memoryplan_prompt_path}")
        raise e

def extract_memory_notes_prompt() -> dict:
    '''
    从用户query种抽去事实性的记忆,但又不用存储在用户画像的内容
    '''
    try:
        extract_memory_notes_prompt_path = get_abs_path(prompts_config["extract_memory_notes_prompt_path"])
    except KeyError as e:
        log.error(f"not found report_prompt_path please check your config:{prompts_config}")
        raise e

    try:
        with open(extract_memory_notes_prompt_path,"r",encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as e:
        log.error(f"not found system prompt file:{extract_memory_notes_prompt_path}")
        raise e

def classify_fact_spans_prompt() -> dict:
    '''
    从用户query种抽去事实性的记忆,但又不用存储在用户画像的内容
    '''
    try:
        classify_fact_spans_path = get_abs_path(prompts_config["classify_fact_spans_path"])
    except KeyError as e:
        log.error(f"not found report_prompt_path please check your config:{prompts_config}")
        raise e

    try:
        with open(classify_fact_spans_path,"r",encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as e:
        log.error(f"not found system prompt file:{classify_fact_spans_path}")
        raise e

def memory_consolidator_prompt() -> dict:
    '''
    从用户query种抽去事实性的记忆,但又不用存储在用户画像的内容
    '''
    try:
        memory_consolidator_prompt_path = get_abs_path(prompts_config["memory_consolidator_prompt_path"])
    except KeyError as e:
        log.error(f"not found report_prompt_path please check your config:{prompts_config}")
        raise e

    try:
        with open(memory_consolidator_prompt_path,"r",encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as e:
        log.error(f"not found system prompt file:{memory_consolidator_prompt_path}")
        raise e


def memory_note_filter_prompt() -> str:
    try:
        memory_note_filter_prompt_path = get_abs_path(prompts_config["memory_note_filter_prompt_path"])
    except KeyError as e:
        log.error(f"not found memory_note_filter_prompt_path in prompts config: {prompts_config}")
        raise e

    try:
        with open(memory_note_filter_prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as e:
        log.error(f"not found memory note filter prompt file: {memory_note_filter_prompt_path}")
        raise e

def router_query_prompt() -> str:
    try:
        ROUTE_QUERY_PROMPT_PATH = get_abs_path(prompts_config["ROUTE_QUERY_PROMPT_PATH"])
    except KeyError as e:
        log.error(f"not found memory_note_filter_prompt_path in prompts config: {prompts_config}")
        raise e

    try:
        with open(ROUTE_QUERY_PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as e:
        log.error(f"not found memory note filter prompt file: {ROUTE_QUERY_PROMPT_PATH}")
        raise e

def load_rag_answer_prompt() -> str:
    try:
        rag_answer_prompt_path = get_abs_path(prompts_config["rag_answer_prompt_path"])
    except KeyError as e:
        log.error(f"not found rag_answer_prompt_path in prompts config: {prompts_config}")
        raise e

    try:
        with open(rag_answer_prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as e:
        log.error(f"not found rag answer prompt file: {rag_answer_prompt_path}")
        raise e

def load_report_writer_prompt() -> str:
    try:
        report_writer_prompt_path = get_abs_path(prompts_config["report_writer_prompt_path"])
    except KeyError as e:
        log.error(f"not found report_writer_prompt_path in prompts config: {prompts_config}")
        raise e

    try:
        with open(report_writer_prompt_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError as e:
        log.error(f"not found report writer prompt file: {report_writer_prompt_path}")
        raise e


if __name__ == '__main__':
    print(load_system_prompt())