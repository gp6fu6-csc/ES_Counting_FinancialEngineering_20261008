import importlib.util
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "check_licenses", Path(__file__).resolve().parent.parent / "scripts" / "check_licenses.py")
check_licenses = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_licenses)


class ClassifyTests(unittest.TestCase):
    def test_permissive_licences_pass(self):
        for text in ("MIT", "MIT License", "MIT-CMU", "BSD License", "BSD-3-Clause", "Apache-2.0",
                     "Apache Software License; BSD License", "Apache-2.0 OR BSD-2-Clause",
                     "Python Software Foundation License", "ISC License (ISCL)",
                     "GNU Lesser General Public License v3 (LGPLv3)", "LGPL-2.1-or-later"):
            self.assertTrue(check_licenses.classify(text)[0], text)

    def test_restrictive_or_unknown_licences_fail(self):
        for text in ("GPL-3.0", "GNU General Public License v3 (GPLv3)", "AGPL-3.0", "SSPL-1.0",
                     "Other/Proprietary License", "Commercial", "CC-BY-NC-4.0", "UNKNOWN", "",
                     "MIT; GNU General Public License v2 (GPLv2)", "Some custom licence"):
            self.assertFalse(check_licenses.classify(text)[0], text)


if __name__ == "__main__":
    unittest.main()
