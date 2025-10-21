import threading
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, RedirectResponse
from pathlib import Path
import subprocess
import shutil


class Editor:
    def __init__(self, url: str="0.0.0.0", port: int=3000, connect_to:str ="0.0.0.0", connect_port:int=8000, verbose: bool=False):
        self.host = url
        self.port = port
        self.connect_to = connect_to
        self.connect_port = connect_port
        self.verbose = verbose
        self.app = FastAPI()

        self.repo_url = "https://github.com/RFDigitalTwin/LocalEditor.git"
        self.base_dir = Path.cwd() / ".rfdt" / "LocalEditor"
        self._update_frontend()

        self.app.mount("/", StaticFiles(directory=str(self.base_dir), html=True), name="static")
        @self.app.get("/environment.exr")
        async def get_environment():
            return FileResponse(str(self.base_dir) / "environment.exr", media_type="image/exr")
        @self.app.get("/")
        async def root():
            return RedirectResponse(url="/")
        self.thread = None


    def _update_frontend(self):
        if shutil.which("git") is None:
            if self.verbose:
                print("⚠️ Warning: 'git' command not found. Please install Git to enable frontend update.")
            return
        if not self.base_dir.exists():
            if self.verbose:
                print("Cloning Editor repo...")
            self.base_dir.parent.mkdir(parents=True, exist_ok=True)
            # Use DEVNULL to suppress git output in non-verbose mode
            stdout = None if self.verbose else subprocess.DEVNULL
            stderr = None if self.verbose else subprocess.DEVNULL
            subprocess.run(["git", "clone", self.repo_url, str(self.base_dir)], 
                         check=True, stdout=stdout, stderr=stderr)
        else:
            if self.verbose:
                print("Updating Editor repo...")
            stdout = None if self.verbose else subprocess.DEVNULL
            stderr = None if self.verbose else subprocess.DEVNULL
            subprocess.run(["git", "-C", str(self.base_dir), "fetch"], 
                         check=True, stdout=stdout, stderr=stderr)
            local_hash = subprocess.check_output(
                ["git", "-C", str(self.base_dir), "rev-parse", "HEAD"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            remote_hash = subprocess.check_output(
                ["git", "-C", str(self.base_dir), "rev-parse", "origin/main"],
                stderr=subprocess.DEVNULL
            ).decode().strip()
            if local_hash != remote_hash:
                if self.verbose:
                    print("New version detected, pulling updates...")
                subprocess.run(["git", "-C", str(self.base_dir), "pull"], 
                             check=True, stdout=stdout, stderr=stderr)
            else:
                if self.verbose:
                    print("Editor repo already up-to-date.")

    def start(self):
        if self.thread is None:
            # Configure uvicorn logging based on verbose setting
            log_level = "info" if self.verbose else "error"
            self.thread = threading.Thread(
                target=lambda: uvicorn.run(
                    self.app, 
                    host=self.host, 
                    port=self.port,
                    log_level=log_level,
                    access_log=self.verbose
                ),
                daemon=True
            )
            self.thread.start()
            if self.verbose:
                print(f"Editor server started at http://{self.host}:{self.port}/Editor")


    def stop(self):
        self.thread.join()