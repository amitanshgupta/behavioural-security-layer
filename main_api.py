import uvicorn
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

if __name__ == "__main__":
    uvicorn.run(
        "api.main_api:app",
        host       = "0.0.0.0",
        port       = 8000,
        reload     = True,
        log_level  = "info",
    )