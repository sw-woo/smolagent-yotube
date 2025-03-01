# 필요한 라이브러리 가져오기
import datasets  # 🤖 데이터셋을 로드하기 위한 라이브러리
from langchain.docstore.document import Document  # 📝 문서 형식으로 데이터를 다루기 위한 라이브러리
from langchain.text_splitter import RecursiveCharacterTextSplitter  # 📄 텍스트를 분할하기 위한 라이브러리
from langchain_community.retrievers import BM25Retriever  # 🔍 BM25 검색 알고리즘을 활용한 검색기(retriever)
from smolagents import Tool  # 🛠️ 도구(Tool) 생성 및 관리
from smolagents import HfApiModel, CodeAgent  # 🤖 코드 에이전트와 Hugging Face API 모델
from smolagents import CodeAgent, LiteLLMModel  # 🧠 가벼운 LLM 모델과 코드 에이전트

# 💾 Hugging Face의 transformers 문서 데이터셋을 로드
knowledge_base = datasets.load_dataset("m-ric/huggingface_doc", split="train")

# 🔍 transformers 관련 문서만 필터링
knowledge_base = knowledge_base.filter(lambda row: row["source"].startswith("huggingface/transformers"))

# 📄 문서를 리스트로 변환 (출처 정보 포함)
source_docs = [
    Document(page_content=doc["text"], metadata={"source": doc["source"].split("/")[1]})
    for doc in knowledge_base
]

# 📚 텍스트를 분할하는 설정 (문서 검색의 성능을 높이기 위함)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # 📏 한 조각의 최대 길이 (500자)
    chunk_overlap=50,  # 🔄 조각 간의 겹치는 부분 (50자)
    add_start_index=True,  # 🔢 시작 인덱스 추가
    strip_whitespace=True,  # 🚀 불필요한 공백 제거
    separators=["\n\n", "\n", ".", " ", ""],  # 📍 텍스트를 분할할 기준
)

# 📜 문서를 작은 단위로 나누기
docs_processed = text_splitter.split_documents(source_docs)


# 🔧 검색 기능을 위한 사용자 정의 도구(Tool) 클래스
class RetrieverTool(Tool):
    name = "retriever"  # 🏷️ 도구의 이름
    description = "transformers 문서에서 관련 내용을 검색하는 기능"  # 📜 도구의 설명
    inputs = {
        "query": {
            "type": "string",
            "description": "검색할 내용. 가능한 한 구체적으로 작성해야 합니다.",
        }
    }
    output_type = "string"  # 📄 출력 형식 (문자열)

    def __init__(self, docs, **kwargs):
        super().__init__(**kwargs)
        self.retriever = BM25Retriever.from_documents(
            docs, k=10  # 🔍 최대 10개의 관련 문서를 검색
        )

    # 🔎 검색 실행
    def forward(self, query: str) -> str:
        assert isinstance(query, str), "검색어는 문자열이어야 합니다!"

        docs = self.retriever.invoke(query)  # 🔍 검색 수행
        return "\n검색된 문서:\n" + "".join(
            [
                f"\n\n===== 문서 {str(i)} =====\n" + doc.page_content
                for i, doc in enumerate(docs)
            ]
        )


# 🛠️ 검색 도구 인스턴스 생성
retriever_tool = RetrieverTool(docs_processed)

# 🤖 사용할 LLM 모델 설정
model = LiteLLMModel(
    model_id="ollama_chat/llama3.2:3b",  # 🎯 사용할 모델 (3B 크기의 Llama3.2)
    api_base="http://localhost:11434",  # 🌐 로컬 서버에서 API 실행
    num_ctx=8192,  # 📏 컨텍스트 길이 (기본값 2048보다 크게 설정하여 문맥 유지)
)

# 🧠 코드 실행 에이전트(CodeAgent) 생성
agent = CodeAgent(
    tools=[retriever_tool],  # 🛠️ 도구 추가 (검색 기능 포함)
    model=model,  # 🤖 모델 지정
    add_base_tools=True,  # 🔧 기본 도구 포함 여부 (예: 계산기, 파일 관리 등)
    additional_authorized_imports=['numpy', 'sys', 'wikipedia', 'requests', 'bs4']  # 📚 허용된 추가 라이브러리
)

# 🏃‍♂️ 에이전트를 실행하여 transformers 모델 훈련 속도 관련 질문을 처리
agent_output = agent.run("트랜스포머 모델을 훈련할 때, 순전파(forward)와 역전파(backward) 중 어떤 것이 더 느린가요?")

# 📢 최종 결과 출력
print("최종 출력:")
print(agent_output)