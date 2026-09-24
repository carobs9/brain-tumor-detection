import argparse
import os
from pathlib import Path

import kagglehub
from dotenv import load_dotenv


DATASET = "sartajbhuvaji/brain-tumor-classification-mri"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the brain tumor dataset.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            os.environ.get(
                "BRAIN_TUMOR_DATA_DIR",
                os.environ.get("DATA_ROOT", PROJECT_ROOT / "data" / "raw"),
            )
        ),
        help="Directory where Training/ and Testing/ are stored.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    path = kagglehub.dataset_download(DATASET, output_dir=str(output_dir))
    print("Path to dataset files:", path)


if __name__ == "__main__":
    main()