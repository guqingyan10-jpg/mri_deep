"""Importing lesion metrics must not require notebook visualization packages."""

import subprocess
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class EvaluationPackageImportTest(unittest.TestCase):
    def test_visualization_is_lazy_and_public_export_still_works(self):
        script = (
            "import evaluation, sys, types\n"
            "assert 'evaluation.visualization' not in sys.modules\n"
            "assert 'evaluation.evaluator' not in sys.modules\n"
            "assert 'evaluation.advanced_metrics' not in sys.modules\n"
            "fake = types.ModuleType('evaluation.visualization')\n"
            "fake.ShowResult = object()\n"
            "sys.modules['evaluation.visualization'] = fake\n"
            "assert evaluation.ShowResult is fake.ShowResult\n"
            "assert evaluation.ShowResult is fake.ShowResult\n"
        )
        subprocess.run([sys.executable, "-c", script], cwd=ROOT, check=True)


if __name__ == "__main__":
    unittest.main()
