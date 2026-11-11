# API test — /v1/chat/completions returns an OpenAI-compatible shape.
#
# Run: hurl --test --variable host=http://127.0.0.1:11435 --variable model=<id> \
#            hurl/api/chat_completions.hurl

POST {{host}}/v1/chat/completions
Content-Type: application/json
{
  "model": "{{model}}",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 5,
  "temperature": 0
}
HTTP 200
[Asserts]
header "Content-Type" contains "application/json"
jsonpath "$.id" exists
jsonpath "$.object" == "chat.completion"
jsonpath "$.model" exists
jsonpath "$.choices" count >= 1
jsonpath "$.choices[0].message.role" == "assistant"
jsonpath "$.choices[0].message.content" isString
jsonpath "$.choices[0].finish_reason" exists
jsonpath "$.usage.prompt_tokens" >= 1
jsonpath "$.usage.completion_tokens" >= 1
