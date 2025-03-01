from smolagents import CodeAgent, LiteLLMModel  # 🧠 가벼운 LLM 모델과 코드 에이전트

# 🤖 사용할 LLM 모델 설정
model = LiteLLMModel(
    model_id="ollama_chat/llama3.2:3b",  # 🎯 사용할 모델 (3B 크기의 Llama3.2)
    api_base="http://localhost:11434",  # 🌐 로컬 서버에서 API 실행
    num_ctx=8192,  # 📏 컨텍스트 길이 (기본값 2048보다 크게 설정하여 문맥 유지)
)

# 🧠 코드 실행 에이전트(CodeAgent) 생성
agent = CodeAgent(
    tools=[],  # 🛠️ 도구 추가 (검색 기능 포함)
    model=model,  # 🤖 모델 지정
    add_base_tools=True,  # 🔧 기본 도구 포함 여부 (예: 계산기, 파일 관리 등)
    additional_authorized_imports=['numpy', 'sys', 'wikipedia', 'requests', 'bs4']  # 📚 허용된 추가 라이브러리
)

# 🏃‍♂️ 에이전트를 실행하여 transformers 모델 훈련 속도 관련 질문을 처리
agent_output = agent.run("트랜스포머 모델을 훈련할 때, 순전파(forward)와 역전파(backward) 중 어떤 것이 더 느린가요?")

# 📢 최종 결과 출력
print("최종 출력:")
print(agent_output)