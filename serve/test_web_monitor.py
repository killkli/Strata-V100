"""The dashboard must bootstrap using the HTML actually served (a merge mixed two layouts)."""
import os
import shutil
import subprocess
import unittest
from pathlib import Path

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, serve


class ReloadableEngine(MockEngine):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, "Ready.</think>\nHello.", max_context=4096)
        self.running = True

    def alive(self):
        return self.running

    def unload(self):
        self.running = False

    def restart(self):
        self.running = True


@unittest.skipUnless(os.environ.get("WEBTEST_MODULES") and shutil.which("node"),
                     "set WEBTEST_MODULES and CHROME_BIN to run the monitor in a browser")
class MonitorInBrowser(unittest.TestCase):
    def test_bootstrap_and_live_dashboard(self):
        root = Path(__file__).resolve().parents[1]
        tok = ByteTokenizer()
        svc = Service(ReloadableEngine(tok), tok, ChatTemplate(root / "serve/chat_template.jinja"))
        svc.api_key = "monitor-browser-test-key"
        httpd = serve(svc, port=0)
        try:
            env = {**os.environ, "WEBTEST_SERVER": f"http://127.0.0.1:{httpd.server_address[1]}",
                   "WEBTEST_KEY": svc.api_key}
            result = subprocess.run(["node", str(root / "tools/webtest/browser_test.mjs"), "monitor:"],
                                    capture_output=True, text=True, timeout=60, env=env)
        finally:
            httpd.shutdown()
            httpd.server_close()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
