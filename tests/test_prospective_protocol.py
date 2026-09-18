import copy
import json
from pathlib import Path
import tempfile
import unittest

from etas_challenge.prospective_protocol import validate_protocol


class ProspectiveProtocolTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.path = self.root / "configs/prospective/three-region-dry-run-v1.json"

    def test_locked_three_region_protocol(self):
        protocol = validate_protocol(self.path, self.root)
        self.assertEqual(
            [region["region_id"] for region in protocol["regions"]],
            ["california-relm", "new-zealand-csep", "chile-subduction"],
        )
        self.assertFalse(protocol["counts_toward_prospective_claim"])

    def test_japan_cannot_enter_protocol(self):
        protocol = json.loads(self.path.read_text(encoding="utf-8"))
        changed = copy.deepcopy(protocol)
        changed["regions"][0]["region_id"] = "japan-a"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as stream:
            json.dump(changed, stream)
            temporary = Path(stream.name)
        self.addCleanup(temporary.unlink)
        with self.assertRaisesRegex(ValueError, "exactly the three admitted regions"):
            validate_protocol(temporary, self.root)

    def test_formal_protocol_is_frozen_for_365_days(self):
        path = (
            self.root
            / "configs/prospective/ch008-three-region-prospective-v1.json"
        )
        protocol = validate_protocol(path, self.root)
        self.assertEqual(protocol["mode"], "prospective")
        self.assertEqual(protocol["duration_days"], 365)
        self.assertEqual(protocol["minimum_events"], 500)
        self.assertTrue(protocol["counts_toward_prospective_claim"])
        self.assertEqual(
            protocol["automatic_activation"]["activation_issue_date_utc"],
            "2026-09-23",
        )

    def test_multiregion_spatial_protocol_is_formal_and_has_four_regions(self):
        path = self.root / "configs/prospective/multi-region-spatial-etas-prospective-v1.json"
        protocol = validate_protocol(path, self.root)
        self.assertEqual(protocol["mode"], "prospective")
        self.assertEqual(protocol["duration_days"], 365)
        self.assertEqual(
            {region["region_id"] for region in protocol["regions"]},
            {"california-relm", "new-zealand-csep", "chile-subduction", "japan-c"},
        )
        self.assertEqual(protocol["csep_evaluation"]["tests"], ["N-test", "L-test", "R-test"])


if __name__ == "__main__":
    unittest.main()
