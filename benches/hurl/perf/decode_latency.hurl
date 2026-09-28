# Performance gate — decode latency.
# A short response must arrive within a generous CI threshold (not a precision
# bench — that's what nxm-bench is for).
#
# Run: hurl --test --variable host=http://127.0.0.1:11435 --variable model=<id> \
#            hurl/perf/decode_latency.hurl

POST {{host}}/v1/chat/completions
Content-Type: application/json
{
  "model": "{{model}}",
  "messages": [{"role": "user", "content": "Hi"}],
  "max_tokens": 10,
  "temperature": 0
}
HTTP 200
[Asserts]
jsonpath "$.choices[0].message.content" isString
duration < 5000
