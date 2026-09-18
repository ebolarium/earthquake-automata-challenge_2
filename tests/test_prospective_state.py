from datetime import datetime, timezone
import unittest

import numpy as np

from etas_challenge.prospective_state import BootstrapCatalog
from etas_challenge.prospective_state import catalog_history_sha256
from etas_challenge.prospective_state import model_state_id
from etas_challenge.prospective_state import validate_state_artifact


class ProspectiveStateTest(unittest.TestCase):
    def test_state_identity_is_stable_and_boundary_sensitive(self):
        as_of = datetime(2026, 8, 19, tzinfo=timezone.utc)
        cutoff = datetime(2026, 8, 30, tzinfo=timezone.utc)
        first = model_state_id(
            "protocol", "region", as_of, cutoff, "c" * 64, "a" * 64, "b" * 64
        )
        second = model_state_id(
            "protocol", "region", as_of, cutoff, "c" * 64, "a" * 64, "b" * 64
        )
        changed = model_state_id(
            "protocol", "region", datetime(2026, 8, 20, tzinfo=timezone.utc),
            cutoff, "c" * 64, "a" * 64, "b" * 64,
        )
        self.assertEqual(first, second)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(first), 64)

    def test_catalog_identity_is_stable_and_event_sensitive(self):
        catalog = BootstrapCatalog(
            (1,),
            np.array(["event-a", "event-b"]),
            np.array([1, 2], dtype=np.int64),
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([5.0, 6.0]),
            np.array([2.5, 3.0]),
        )
        first = catalog_history_sha256(catalog)
        same_events_other_snapshot = BootstrapCatalog(
            (99,), catalog.event_ids, catalog.origin_time_ns, catalog.latitudes,
            catalog.longitudes, catalog.depths_km, catalog.magnitudes,
        )
        self.assertEqual(first, catalog_history_sha256(same_events_other_snapshot))
        changed = BootstrapCatalog(
            (1,), catalog.event_ids, catalog.origin_time_ns, catalog.latitudes,
            catalog.longitudes, catalog.depths_km, np.array([2.5, 3.1]),
        )
        self.assertNotEqual(first, catalog_history_sha256(changed))

    def test_regional_state_artifact_contract(self):
        as_of = "2026-08-19T00:00:00+00:00"
        catalog = BootstrapCatalog(
            (1,),
            np.array(["event-a", "event-b"]),
            np.array([1, 2], dtype=np.int64),
            np.array([1.0, 2.0]),
            np.array([3.0, 4.0]),
            np.array([5.0, 6.0]),
            np.array([2.5, 3.0]),
        )
        arrays = {
            "event_ids": catalog.event_ids,
            "origin_time_ns": catalog.origin_time_ns,
            "latitudes": catalog.latitudes,
            "longitudes": catalog.longitudes,
            "depths_km": catalog.depths_km,
            "magnitudes": catalog.magnitudes,
            "event_etas_rates": np.array([0.2, 0.3]),
            "event_background_probabilities": np.array([0.5, 0.4]),
            "event_cells": np.array([0, 1]),
            "ch008_age": np.array([1.0, 2.0]),
            "ch008_exposure": np.array([3.0, 4.0]),
            "ch008_roots": np.array([0.1, 0.2]),
            "as_of": np.asarray(as_of),
            "catalog_cutoff": np.asarray("2026-08-30T00:00:00+00:00"),
            "etas_model_sha256": np.asarray("a" * 64),
            "ch008_model_sha256": np.asarray("b" * 64),
        }
        manifest = {
            "as_of": as_of,
            "catalog_cutoff": "2026-08-30T00:00:00+00:00",
            "events": 2,
            "snapshot_ids": [1],
            "catalog_history_sha256": catalog_history_sha256(catalog),
            "baseline_model_sha256": "a" * 64,
            "challenger_model_sha256": "b" * 64,
        }
        with self._npz(arrays) as archive:
            result = validate_state_artifact(
                archive, manifest, expected_state_shape=(2,), regional=True
            )
        self.assertEqual(result["events"], 2)
        self.assertEqual(result["state_shape"], [2])

    def test_evidence_gate_state_tracks_incumbent_and_challenger_separately(self):
        as_of = "2026-08-19T00:00:00+00:00"
        arrays = {
            "event_ids": np.array([], dtype="<U1"),
            "origin_time_ns": np.array([], dtype=np.int64),
            "latitudes": np.array([], dtype=float),
            "longitudes": np.array([], dtype=float),
            "depths_km": np.array([], dtype=float),
            "magnitudes": np.array([], dtype=float),
            "ch008_age": np.array([1.0, 2.0]),
            "ch008_exposure": np.array([3.0, 4.0]),
            "ch008_roots": np.array([0.1, 0.2]),
            "background_root_days": np.array([1, 2], dtype=np.int64),
            "background_root_values": np.ones((2, 3)),
            "gate_log_bayes_factor": np.asarray(0.0),
            "as_of": np.asarray(as_of),
            "catalog_cutoff": np.asarray("2026-09-18T00:00:00+00:00"),
            "etas_model_sha256": np.asarray("a" * 64),
            "ch008_model_sha256": np.asarray("b" * 64),
        }
        manifest = {
            "as_of": as_of,
            "catalog_cutoff": "2026-09-18T00:00:00+00:00",
            "events": 0,
            "snapshot_ids": [],
            "catalog_history_sha256": catalog_history_sha256(BootstrapCatalog(
                (), arrays["event_ids"], arrays["origin_time_ns"], arrays["latitudes"],
                arrays["longitudes"], arrays["depths_km"], arrays["magnitudes"],
            )),
            "baseline_model_sha256": "a" * 64,
            "challenger_model_sha256": "c" * 64,
        }
        with self._npz(arrays) as archive:
            result = validate_state_artifact(
                archive, manifest, expected_state_shape=(2,), regional=False,
                evidence_gate=True, incumbent_model_sha256="b" * 64,
            )
        self.assertEqual(result["events"], 0)

    @staticmethod
    def _npz(arrays):
        import contextlib
        import io

        buffer = io.BytesIO()
        np.savez(buffer, **arrays)
        buffer.seek(0)
        return contextlib.closing(np.load(buffer, allow_pickle=False))


if __name__ == "__main__":
    unittest.main()
