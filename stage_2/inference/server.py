
# This file provides the HTTP entry point for the CUDA text-inference service.
# Access point: run it directly; it listens on --host/--port (127.0.0.1:8000 by default).
# It delivers health/model JSON plus streamed or complete text-completion results.
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import queue
import time
from backend import CudaBackend, Tokenizer
from scheduler import Busy, Worker


def handler(worker, model):
    # Create an HTTP handler bound to the shared scheduler worker.
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def reply(self, code, data):
            # Send JSON with a matching Content-Length header.
            body = json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers(); self.wfile.write(body)

        def do_GET(self):
            # Expose health and model discovery through read-only endpoints.
            if self.path == "/health":
                self.reply(200 if worker.thread.is_alive() else 503, {"ready": worker.thread.is_alive()})
            elif self.path == "/v1/models":
                self.reply(200, {"object": "list", "data": [{"id": model, "object": "model"}]})
            else:
                self.reply(404, {"error": "Unknown endpoint"})

        def do_POST(self):
            # Track request cancellation and streaming status across completion handling.
            ticket = None
            streaming_started = False
            try:
                # Route and size-check the single completion endpoint.
                if self.path != "/v1/completions":
                    self.reply(404, {"error": "Unknown endpoint"}); return
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 1024*1024:
                    raise ValueError("Request body must be between 1 byte and 1 MiB")
                self.connection.settimeout(30)

                # Decode the JSON object and reject malformed request bodies.
                request = json.loads(self.rfile.read(size))
                if not isinstance(request, dict):
                    raise ValueError("Expected a JSON object")

                # Validate supported fields and boolean options.
                allowed = {"model", "prompt", "max_tokens", "temperature", "n", "beam_width",
                           "seed", "stream", "stream_options", "ignore_eos", "skip_special_tokens"}
                if set(request)-allowed:
                    raise ValueError(f"Unsupported parameters: {sorted(set(request)-allowed)}")
                if request.get("model", model) != model:
                    raise ValueError("Unknown model")
                for name in ("stream", "ignore_eos", "skip_special_tokens"):
                    if name in request and not isinstance(request[name], bool):
                        raise ValueError(f"{name} must be a boolean")

                # Validate stream options and beam-search compatibility.
                options = request.get("stream_options", {})
                if not isinstance(options, dict) or set(options)-{"include_usage"}:
                    raise ValueError("Only stream_options.include_usage is supported")
                if request.get("beam_width", 1) > 1 and request.get("stream"):
                    raise ValueError("Beam search requires stream=false because the winning prefix can change")

                # Submit work and assign a response ID and creation timestamp.
                ticket = worker.submit(request)
                ident, created = ticket.ident, int(time.time())

                def envelope(choices, usage=None):
                    # Wrap choices in an OpenAI-compatible text-completion envelope.
                    result = dict(id=ident, object="text_completion", created=created,
                                  model=model, choices=choices)
                    if usage is not None:
                        result["usage"] = usage
                    return result

                def send(data):
                    # Write and immediately flush one Server-Sent Events frame.
                    self.wfile.write(("data: "+json.dumps(data, ensure_ascii=False)+"\n\n").encode())
                    self.wfile.flush()

                if request.get("stream"):
                    # Wait for validation or the first output before committing HTTP 200.
                    pending = None
                    while not ticket.future.done():
                        try:
                            pending = ticket.events.get(timeout=0.1); break
                        except queue.Empty:
                            pass
                    if ticket.future.done():
                        ticket.future.result()

                    # Configure the HTTP response as an event stream.
                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("X-Accel-Buffering", "no")
                    self.send_header("Connection", "close")
                    self.end_headers(); streaming_started = True

                    # Stream generated deltas while tracking text already delivered.
                    sent = {}
                    while pending is not None or not ticket.future.done() or not ticket.events.empty():
                        if pending is not None:
                            event, pending = pending, None
                        else:
                            try:
                                event = ticket.events.get(timeout=0.1)
                            except queue.Empty:
                                continue
                        sent[event["index"]] = sent.get(event["index"], "")+event["text"]
                        send(envelope([event]))

                    # Send final suffixes, optional usage, and stream termination.
                    result = ticket.future.result()
                    for choice in result["choices"]:
                        prior = sent.get(choice["index"], "")
                        send(envelope([dict(index=choice["index"], text=choice["text"][len(prior):],
                                            finish_reason=choice["finish_reason"])]))
                    if request.get("stream_options", {}).get("include_usage"):
                        send(envelope([], usage(result)))
                    self.wfile.write(b"data: [DONE]\n\n"); self.wfile.flush()
                    self.close_connection = True
                else:
                    # Wait for the full result and return one JSON response.
                    result = ticket.future.result(timeout=3600)
                    choices = [{k: c[k] for k in ("index", "text", "finish_reason")} for c in result["choices"]]
                    self.reply(200, envelope(choices, usage(result)))
            except Busy as exc:
                # Return HTTP 429 when the scheduler has no request capacity.
                self.reply(429, {"error": str(exc)})
            except (BrokenPipeError, ConnectionResetError, TimeoutError):
                # Close the connection quietly after client-side breaks or timeouts.
                self.close_connection = True
            except Exception as exc:
                # Stream an error if headers were sent; otherwise return an HTTP JSON error.
                if streaming_started:
                    try:
                        send({"error": str(exc)})
                    except OSError:
                        pass
                    self.close_connection = True
                else:
                    self.reply(400 if isinstance(exc, (ValueError, KeyError, TypeError)) else 503,
                               {"error": str(exc)})
            finally:
                # Mark outstanding work cancelled when the client disconnects.
                if ticket and not ticket.future.done():
                    ticket.cancelled.set()

        def log_message(self, *args):
            # Suppress default per-request access logging.
            pass
    return Handler


def usage(result):
    # Aggregate prompt and generated-token counts for API usage reporting.
    return dict(prompt_tokens=result["prompt_tokens"], completion_tokens=result["completion_tokens"],
                total_tokens=result["prompt_tokens"]+result["completion_tokens"])


def main():
    # Define CLI settings for model paths, execution limits, and server binding.
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--library-dir", type=Path, default=Path(__file__).parent/"build")
    p.add_argument("--mode", choices=["none", "contiguous", "paged"], default="paged")
    p.add_argument("--blocks", type=int, default=1024)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--max-context", type=int, default=8192)
    p.add_argument("--max-rows", type=int, default=256)
    p.add_argument("--max-sequences", type=int, default=32)
    p.add_argument("--max-requests", type=int, default=32)
    p.add_argument("--prefix-capacity", type=int, default=64)
    p.add_argument("--no-batching", action="store_true")
    p.add_argument("--telemetry", type=Path, help="Append iteration/request measurements as JSONL")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)

    # Parse configuration and connect the CUDA backend to the scheduler.
    args = p.parse_args()
    settings = dict(blocks=args.blocks, block_size=args.block_size, max_context=args.max_context,
                    max_rows=args.max_rows, max_sequences=args.max_sequences)
    worker = Worker(lambda: CudaBackend(args.model_dir, args.library_dir/"libqwen3_runtime.so", **settings),
                    lambda: Tokenizer(args.library_dir/"libqwen3_tokenizer.so", args.model_dir),
                    **settings, mode=args.mode, batch=not args.no_batching,
                    max_requests=args.max_requests, prefix_capacity=args.prefix_capacity,
                    telemetry_path=args.telemetry)

    # Bind the threaded HTTP server and serve until shutdown.
    server = ThreadingHTTPServer((args.host, args.port), handler(worker, args.model))
    try:
        server.serve_forever()
    finally:
        # Close resources and print final worker statistics on shutdown.
        worker.close(); server.server_close()
        print(json.dumps(worker.stats))


if __name__ == "__main__":
    # Start the service only when this file is executed directly.
    main()
