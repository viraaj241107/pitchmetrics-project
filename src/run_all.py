# Runs every step in order. This is what GitHub Actions runs every week.
#
#   python run_all.py            -> full run (download, train, predict, build site)
#   python run_all.py --no-train -> skip training and reuse the saved models

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent

steps = ["step1_download.py", "step2_build_dataset.py", "step3_train.py", "step4_predict.py",
         "step5_clubs.py", "step6_roles.py", "step7_build_site.py"]
if "--no-train" in sys.argv:
    steps.remove("step3_train.py")

for step in steps:
    print(f"\n===== {step} =====")
    result = subprocess.run([sys.executable, step], cwd=HERE)
    if result.returncode != 0:
        print(f"{step} failed - stopping.")
        sys.exit(1)
print("\nDone. Open docs/index.html to see the website.")
