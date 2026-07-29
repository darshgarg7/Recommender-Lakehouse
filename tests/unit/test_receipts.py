import json
import tempfile
import unittest
from pathlib import Path

from marketplace_recommender.governance.receipts import (
    ReceiptVerificationError,
    verify_run_receipt,
    write_run_receipt,
)


class RunReceiptTests(unittest.TestCase):
    def _write(self, root: Path) -> Path:
        artifact = root / "gold/output.jsonl"
        artifact.parent.mkdir(parents=True)
        artifact.write_text('{"id":1}\n', encoding="utf-8")
        receipt = root / "monitoring/run_receipt.json"
        write_run_receipt(
            receipt,
            run_root=root,
            identity={"tier": "test", "seed": 7},
            source_contract={"reviews": "a" * 64},
            temporal_contract={"historical_join_predicate": "event < label"},
            decision_contract={"serving_champion": "popularity"},
            verified_claims={"rows": 1},
            artifacts={"gold_output": artifact},
        )
        return receipt

    def test_receipt_binds_and_verifies_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = verify_run_receipt(self._write(root), root)
            self.assertTrue(result["valid"])
            self.assertEqual(result["verified_artifacts"], ["gold_output"])

    def test_artifact_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = self._write(root)
            (root / "gold/output.jsonl").write_text('{"id":2}\n', encoding="utf-8")
            with self.assertRaisesRegex(ReceiptVerificationError, "digest mismatch"):
                verify_run_receipt(receipt, root)

    def test_payload_tampering_is_detected_before_artifact_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            receipt = self._write(root)
            envelope = json.loads(receipt.read_text(encoding="utf-8"))
            envelope["payload"]["verified_claims"]["rows"] = 2
            receipt.write_text(json.dumps(envelope), encoding="utf-8")
            with self.assertRaisesRegex(ReceiptVerificationError, "payload digest mismatch"):
                verify_run_receipt(receipt, root)

    def test_writer_rejects_an_artifact_outside_the_run_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            parent = Path(temporary)
            root = parent / "run"
            root.mkdir()
            outside = parent / "outside.json"
            outside.write_text("{}\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "inside the run root"):
                write_run_receipt(
                    root / "receipt.json",
                    run_root=root,
                    identity={"tier": "test"},
                    source_contract={"reviews": "a" * 64},
                    temporal_contract={},
                    decision_contract={},
                    verified_claims={},
                    artifacts={"outside": outside},
                )


if __name__ == "__main__":
    unittest.main()
