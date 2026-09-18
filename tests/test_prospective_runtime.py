import unittest

from etas_challenge.prospective_runtime import DRY_RUN_PROTOCOL_ID
from etas_challenge.prospective_runtime import PROSPECTIVE_PROTOCOL_ID
from etas_challenge.prospective_runtime import active_protocol_id
from etas_challenge.prospective_runtime import protocol_path


class Result:
    def __init__(self, row):
        self.row = row

    def fetchone(self):
        return self.row


class Connection:
    def __init__(self, row):
        self.row = row

    def execute(self, query):
        self.query = query
        return Result(self.row)


class ProspectiveRuntimeTest(unittest.TestCase):
    def test_formal_protocol_is_default_for_fresh_multiregion_deployment(self):
        self.assertEqual(active_protocol_id(Connection(None)), PROSPECTIVE_PROTOCOL_ID)

    def test_active_formal_protocol_is_selected(self):
        self.assertEqual(
            active_protocol_id(Connection((PROSPECTIVE_PROTOCOL_ID,))),
            PROSPECTIVE_PROTOCOL_ID,
        )

    def test_protocol_paths_are_distinct(self):
        self.assertNotEqual(
            protocol_path(DRY_RUN_PROTOCOL_ID),
            protocol_path(PROSPECTIVE_PROTOCOL_ID),
        )


if __name__ == "__main__":
    unittest.main()
