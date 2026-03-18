from langchain_core.runnables import RunnableLambda

from model.rerank_client import SiliconRerankModel


def build_reranker_adapter(reranker_model:SiliconRerankModel):
    def reranker_adapter(input_data:dict):
        '''
        因为 RunnableLambda 会把一个普通 Python 函数包装成具备这些能力的对象：
        .invoke()
        .batch()
        .stream()（取决于函数特性）
        可参与 | 管道组合
        也就是说，你不用改底层 client，只要做一层函数包装。
        :param input_data:
        :return:
        '''
        query = input_data["query"]
        docs = input_data["docs"]

        if not docs:
            return {
                "query":query,
                "docs":[]
            }

        reranked_doc = reranker_model.rerank_documents(query,docs)

        return {
            "query":query,
            "docs":reranked_doc
        }

    return RunnableLambda(func=reranker_adapter)


