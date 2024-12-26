import openai

# OpenAI API 키 설정
openai.api_key = 'your-api-key-here'

# OpenAI API 호출 함수
def call_openai_api(prompt):
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=100
    )
    return response.choices[0].text.strip()

# 테스트할 프롬프트
test_prompt = "Hello, how are you?"

# API 호출 및 결과 출력
result = call_openai_api(test_prompt)
print("API Response:", result)