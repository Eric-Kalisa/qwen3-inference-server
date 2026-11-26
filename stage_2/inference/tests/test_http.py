from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
import json
import threading
import unittest
from server import handler
from test_scheduler import worker


class HttpTests(unittest.TestCase):
    def setUp(self):
        self.worker=worker()
        self.server=ThreadingHTTPServer(("127.0.0.1",0),handler(self.worker,"test-model"))
        self.port=self.server.server_address[1]
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True); self.thread.start()
    def tearDown(self):
        self.server.shutdown(); self.server.server_close(); self.thread.join(); self.worker.close()
    def post(self,body):
        c=HTTPConnection("127.0.0.1",self.port,timeout=10)
        c.request("POST","/v1/completions",json.dumps(body),{"Content-Type":"application/json"})
        r=c.getresponse(); status=r.status; raw=r.read(); c.close()
        return status,raw

    def test_stream_matches_nonstream_and_has_usage_and_done(self):
        req=dict(model="test-model",prompt="1 2 3",max_tokens=7,n=3,temperature=.8,seed=42)
        status,raw=self.post(req); expected=json.loads(raw); self.assertEqual(status,200)
        status,raw=self.post(dict(req,stream=True,stream_options={"include_usage":True}))
        self.assertEqual(status,200); self.assertTrue(raw.endswith(b"data: [DONE]\n\n"))
        chunks=[json.loads(x[6:]) for x in raw.decode().split("\n\n") if x.startswith("data: {")]
        texts={i:"" for i in range(3)}; endings={}
        for chunk in chunks:
            for choice in chunk["choices"]:
                texts[choice["index"]]+=choice["text"]
                if choice["finish_reason"]: endings[choice["index"]]=choice["finish_reason"]
        self.assertEqual([texts[i] for i in range(3)],[c["text"] for c in expected["choices"]])
        self.assertEqual(chunks[-1]["usage"],expected["usage"])
        self.assertEqual(expected["usage"]["completion_tokens"],21); self.assertEqual(len(endings),3)

    def test_parallel_http_clients_are_isolated(self):
        requests=[dict(prompt=f"1 2 {i%10}",max_tokens=5) for i in range(24)]
        with ThreadPoolExecutor(8) as ex: results=list(ex.map(self.post,requests))
        self.assertTrue(all(status==200 for status,_ in results))
        ids=[json.loads(raw)["id"] for _,raw in results]; self.assertEqual(len(ids),len(set(ids)))

    def test_invalid_requests_return_errors_before_streaming(self):
        for req in (dict(prompt=""),dict(prompt="1",model="wrong"),dict(prompt="1",stream=True,beam_width=3),
                    dict(prompt="1",max_tokens=9999),dict(prompt="1",unknown=True)):
            status,raw=self.post(req); self.assertEqual(status,400); self.assertIn("error",json.loads(raw))
        status,_=self.post(dict(prompt="1",max_tokens=1)); self.assertEqual(status,200)


if __name__=="__main__": unittest.main()
