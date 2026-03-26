from model.model_factory import chat_model
from pydantic import BaseModel,Field
from utils.prompts_loader import extract_memory_notes_prompt,classify_fact_spans_prompt
from typing import Literal
from utils.logger_handler import logger as log

class CandidateNote(BaseModel):
    text:str = Field(description="适合长期保存的一条事实行记忆,只包含一个事实")
    keywords:list[str] = Field(default=[],description="用于检索的简短的keywords")
    category:Literal["home","device","preference","demand","other"]

class CandidateNoteList(BaseModel):
    notes:list[CandidateNote] = Field(default= [],description="提取出来的长期记忆候选列表")

class FactSpanList(BaseModel):
    spans:list[str] = Field(default=[],description="提取出来的长文本片段")

def split_into_fact_spans(query:str):
    structured_llm =chat_model.with_structured_output(FactSpanList)

    prompt = f"{classify_fact_spans_prompt()} \n\n 用户输入:{query}"
    return structured_llm.invoke(prompt)

def classify_fact_spans(spans: list[str]) -> CandidateNoteList:
    if not spans:
        return CandidateNoteList(notes=[])

    structured_llm = chat_model.with_structured_output(CandidateNoteList)

    prompt = f"{extract_memory_notes_prompt()} \n\n 给定事实片段:{spans}"
    return structured_llm.invoke(prompt)


def extract_candidate_notes(query: str) -> CandidateNoteList:
    span_result = split_into_fact_spans(query)
    note_result:FactSpanList= classify_fact_spans(span_result.spans)

    log.info(f"extract_candidate_notes span_result:{span_result}")
    log.info(f"extract_candidate_notes note_result:{note_result}")

    return note_result


if __name__ == "__main__":
    query = "我朋友家有一个大平台,没有封窗,秋冬天的时候会积灰,而且有点冷,希望有一个机器人能够解决扫地和拖地"
    result = extract_candidate_notes(query)
    print(result)