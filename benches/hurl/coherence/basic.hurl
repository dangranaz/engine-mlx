# Coherence test — basic semantic output.
# These MUST pass after any change to an engine (see nexum RULES).
# If they fail -> corrupted output -> revert immediately.
#
# Run: hurl --test --variable host=http://127.0.0.1:11435 --variable model=<id> \
#            hurl/coherence/basic.hurl

# Test 1: geography (French capital) — explicit question form.
# An explicit question is robust for small quantized *instruct* models; a
# completion-style stub ("The capital of France is") can be misread as chat by a
# 1.7B/4bit model and drift. Assert is unchanged: the answer must name Paris.
POST {{host}}/v1/chat/completions
Content-Type: application/json
{
  "model": "{{model}}",
  "messages": [{"role": "user", "content": "What is the capital of France? Answer with just the city name."}],
  "max_tokens": 20,
  "temperature": 0
}
HTTP 200
[Asserts]
jsonpath "$.choices[0].message.content" matches /[Pp]aris|[Pp]arigi/


# Test 2: arithmetic
POST {{host}}/v1/chat/completions
Content-Type: application/json
{
  "model": "{{model}}",
  "messages": [{"role": "user", "content": "2+2="}],
  "max_tokens": 10,
  "temperature": 0
}
HTTP 200
[Asserts]
jsonpath "$.choices[0].message.content" matches /4/


# Test 3: geography (Italian capital)
POST {{host}}/v1/chat/completions
Content-Type: application/json
{
  "model": "{{model}}",
  "messages": [{"role": "user", "content": "Capital of Italy?"}],
  "max_tokens": 20,
  "temperature": 0
}
HTTP 200
[Asserts]
jsonpath "$.choices[0].message.content" matches /[Rr]om[ae]/
