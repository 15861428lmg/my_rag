import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from openai import OpenAI
from config import (
    MINIMAX_API_KEY,
    MINIMAX_API_BASE,
    MINIMAX_MODEL,
    QWEN_API_KEY,
    QWEN_API_BASE,
    QWEN_EMBEDDING_MODEL,
    CHROMA_PERSIST_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)


class QwenEmbeddings(Embeddings):
    def __init__(self, api_key: str, api_base: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.model = model

    def _parse_embedding_response(self, response):
        if hasattr(response, 'data') and response.data:
            return response.data[0].embedding if hasattr(response.data[0], 'embedding') else response.data[0].get('embedding')
        elif hasattr(response, 'embedding'):
            return response.embedding
        return None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            response = self.client.embeddings.create(
                model=self.model,
                input=text
            )
            embedding = self._parse_embedding_response(response)
            if embedding is None:
                raise ValueError(f"Failed to get embedding for text: {text[:50]}...")
            embeddings.append(embedding)
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=text
        )
        embedding = self._parse_embedding_response(response)
        if embedding is None:
            raise ValueError("Failed to get embedding for query")
        return embedding


class RAGEngine:
    def __init__(self):
        self.embeddings = QwenEmbeddings(
            api_key=QWEN_API_KEY,
            api_base=QWEN_API_BASE,
            model=QWEN_EMBEDDING_MODEL
        )

        self.llm = ChatOpenAI(
            model=MINIMAX_MODEL,
            temperature=0,
            openai_api_key=MINIMAX_API_KEY,
            openai_api_base=MINIMAX_API_BASE,
            extra_body={"reasoning_split": True},
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        self.vectorstore = None
        self.chat_history = []

        self._init_vectorstore()
        self._init_chain()

    def _init_vectorstore(self):
        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings
        )

    def _init_chain(self):
        template = """你是一个专业的文档问答助手。请根据以下提供的上下文信息和对话历史来回答用户的问题。
如果上下文中没有相关信息，请明确告知用户，并尝试基于你的通用知识给出回答，但要说明这不是基于文档的回答。

上下文信息：
{context}

请给出准确、简洁的回答："""

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", template),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])

        self.llm_stream = ChatOpenAI(
            model=MINIMAX_MODEL,
            temperature=0,
            openai_api_key=MINIMAX_API_KEY,
            openai_api_base=MINIMAX_API_BASE,
            streaming=True,
            extra_body={"reasoning_split": True},
        )

        self.rag_chain = (
            self.prompt
            | self.llm_stream
            | StrOutputParser()
        )

        self.rag_chain_non_stream = (
            self.prompt
            | self.llm
            | StrOutputParser()
        )

    def _get_context(self, question: str) -> str:
        docs = self.vectorstore.similarity_search(question, k=4)
        if not docs:
            return "没有找到相关文档内容。"
        return "\n\n".join([doc.page_content for doc in docs])

    def add_documents(self, file_paths: List[str]) -> str:
        all_docs = []

        for file_path in file_paths:
            if not os.path.exists(file_path):
                continue

            ext = os.path.splitext(file_path)[1].lower()

            if ext == '.pdf':
                loader = PyPDFLoader(file_path)
            elif ext == '.docx':
                loader = Docx2txtLoader(file_path)
            elif ext == '.txt':
                loader = TextLoader(file_path, encoding='utf-8')
            else:
                continue

            docs = loader.load()
            all_docs.extend(docs)

        if not all_docs:
            return "没有成功加载任何文档"

        splits = self.text_splitter.split_documents(all_docs)
        self.vectorstore.add_documents(splits)

        return f"成功添加 {len(all_docs)} 个文档，分割为 {len(splits)} 个文本块"

    def query(self, question: str) -> str:
        context = self._get_context(question)

        result = self.rag_chain_non_stream.invoke({
            "question": question,
            "context": context,
            "chat_history": self.chat_history
        })

        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=result))

        return result

    def query_stream(self, question: str):
        context = self._get_context(question)

        full_response = ""
        for chunk in self.rag_chain.stream({
            "question": question,
            "context": context,
            "chat_history": self.chat_history
        }):
            full_response += chunk
            yield full_response

        self.chat_history.append(HumanMessage(content=question))
        self.chat_history.append(AIMessage(content=full_response))

    def clear_memory(self):
        self.chat_history = []

    def clear_vectorstore(self):
        self.vectorstore.delete_collection()
        self._init_vectorstore()
        self.clear_memory()