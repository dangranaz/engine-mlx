# API test — error handling for malformed requests.
# The server must reject bad input with a 4xx status. Different malformed inputs
# map to different (correct) codes with an axum-based server:
#   missing required field -> 422 Unprocessable Entity
#   empty/invalid body     -> 400 Bad Request
#   wrong Content-Type     -> 415 Unsupported Media Type
#
# Run: hurl --test --variable host=http://127.0.0.1:11435 --variable model=<id> \
#            hurl/api/error_handling.hurl

# Missing the required `messages` field -> 422
POST {{host}}/v1/chat/completions
Content-Type: application/json
{
  "model": "{{model}}"
}
HTTP 422

# Empty body -> 400
POST {{host}}/v1/chat/completions
Content-Type: application/json
HTTP 400

# Wrong Content-Type -> 415
POST {{host}}/v1/chat/completions
Content-Type: text/plain
`not json`
HTTP 415
