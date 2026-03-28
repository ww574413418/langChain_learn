'''
这个文件只做 4 件事：

读取 session_notes
读取 global_notes
根据 note_query 做语义召回
做 rerank、去重、top-k 裁剪
注意：
它不负责：
抽取 notes
写 notes
consolidate notes
'''
from pydantic import BaseModel,Field
from memory.memory_note_store import  load_session_notes,load_global_notes
from model.model_factory import embedding_model_silicon,reranker_runnable
import math
from utils.logger_handler import logger as log
from langchain_core.documents import Document

class RetrievedNote(BaseModel):
    text:str = Field(description="note text")
    category:str = Field(description="note category")
    scope:str = Field(description="session or global")
    keywords:list[str] = Field(default_factory=list)
    confidence:float = Field(default=0.0)
    score:float = Field(default=0.0,description="retrival score")

def load_note_candidates(user_id:str|int,thread_id:str|int) ->list[dict]:
    session_notes = load_session_notes(thread_id)
    global_notes = load_global_notes(user_id)
    return session_notes + global_notes


def build_note_search_text(note:dict)->str:
    '''
    将dict转成txt文本,方便embedding
    :param note:
    :return:
    '''
    text = note.get("text","")
    category = note.get("category","")
    keywords = note.get("keywords",[])
    return f"{category}  {keywords}  {text}".strip()

def cosine_similarity(vec1:list[float],vec2:list[float]) ->float:
    '''
    计算cos相似度
    :param vec1:
    :param vec2:
    :return:
    '''
    dot = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a ** 2 for a in vec1))
    norm2 = math.sqrt(sum(b ** 2 for b in vec2))

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

def retrieve_notes_by_embedding(
        user_id:str|int,
        thread_id:str|int,
        note_query:str,
        top_n:int = 10,
)->list[RetrievedNote]:
    candidates = load_note_candidates(user_id,thread_id)
    if not candidates:
        return []

    query_vec = embedding_model_silicon.embed_query(note_query)

    scored_notes = []

    for note in candidates:
        search_text = build_note_search_text(note)
        note_vec = embedding_model_silicon.embed_query(search_text)
        score = cosine_similarity(query_vec,note_vec)

        scored_notes.append(
            RetrievedNote(
                text=note.get("text",""),
                category=note.get("category",""),
                scope=note.get("scope",""),
                keywords=note.get("keywords",[]),
                confidence=note.get("confidence",0.0),
                score=score
            )
        )

    scored_notes.sort(key=lambda x:x.score,reverse=True)
    return scored_notes[:top_n]

def build_rerank_documents(notes:list[RetrievedNote])->list[Document]:
    docs = []

    for note in notes:
        docs.append(
            Document(
                page_content=note.text,
                metadata={
                    "category":note.category,
                    "scope":note.scope,
                    "keywords":note.keywords,
                    "confidence":note.confidence,
                    "embedding_score": note.score
                }
            )
        )

    return docs


def rerank_notes(user_query:str,notes:list[RetrievedNote],top_k:int=5)->list[RetrievedNote]:
    '''
    使用rerank模型对提取出来的notes进行排序
    :param user_query:
    :param notes:
    :param top_k:
    :return:
    '''
    if not notes:
        return []
    # 将list[str] 转成list[Document]
    documents = build_rerank_documents(notes)

    rerank_output = reranker_runnable.invoke(input_data={"query":user_query,"docs":documents})

    rerank_docs = rerank_output["docs"][:top_k]

    reranked_notes = []
    # 重新将list[Document] 转成 list[RetrievedNote]
    for doc in rerank_docs:
        reranked_notes.append(
            RetrievedNote(
                text=doc.page_content,
                category=doc.metadata.get("category",""),
                scope=doc.metadata.get("scope",""),
                keywords=doc.metadata.get("keywords",[]),
                confidence=doc.metadata.get("confidence",0.0),
                score=doc.metadata.get("rerank_score",0.0)
            )
        )

    return reranked_notes

def retrieve_relevant_notes(
        user_id:str|int,
        thread_id:str|int,
        note_query:str,
        top_n:int = 10,
        top_k:int = 5
)->list[RetrievedNote]:
    candidates_notes = retrieve_notes_by_embedding(user_id,thread_id,note_query,top_n)
    reranked_notes = rerank_notes(note_query,candidates_notes,top_k)
    return reranked_notes


def format_retrieved_notes(notes: list[RetrievedNote]) -> str:
    if not notes:
        return ""

    lines = ["当前问题相关的长期记忆片段："]

    for note in notes:
        lines.append(f"- [{note.category}/{note.scope}] {note.text}")

    return "\n".join(lines)


if __name__ == "__main__":
    notes = retrieve_relevant_notes(
        user_id="0001",
        thread_id="123123",
        note_query="适合没有封窗、容易积灰环境的扫拖机器人",
        top_n=8,
        top_k=5,
    )

    for note in notes:
        print(note)

    print(format_retrieved_notes(notes))