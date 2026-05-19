import importlib.util
import unittest
from pathlib import Path


def load_transform_module():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "data_transform" / "transform.py"
    spec = importlib.util.spec_from_file_location("t3_transform", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


transform = load_transform_module()


class TransformInputSchemaTest(unittest.TestCase):
    def test_render_prompt_uses_trace_field(self):
        rendered = transform.render_prompt("Given trace: {trace}", {"trace": "actual"}, 0)

        self.assertEqual(rendered, "Given trace: actual")

    def test_render_prompt_accepts_legacy_text_field(self):
        rendered = transform.render_prompt("Given trace: {trace}", {"text": "legacy"}, 1)

        self.assertEqual(rendered, "Given trace: legacy")

    def test_trace_field_preferred_when_both_fields_are_present(self):
        rendered = transform.render_prompt(
            "Given trace: {trace}",
            {"trace": "published", "text": "legacy"},
            2,
        )

        self.assertEqual(rendered, "Given trace: published")

    def test_missing_or_blank_trace_fails_with_record_index_and_fields(self):
        for record in ({}, {"trace": " ", "text": ""}):
            with self.subTest(record=record):
                with self.assertRaises(ValueError) as context:
                    transform.render_prompt("Given trace: {trace}", record, 12)

                message = str(context.exception)
                self.assertIn("Record 12", message)
                self.assertIn("trace", message)
                self.assertIn("text", message)

    def test_validate_pending_records_ignores_completed_records(self):
        transform.validate_pending_records(
            [{}],
            {"t3_struct": {0}},
            ["t3_struct"],
        )

    def test_validate_pending_records_checks_rows_with_pending_prompt(self):
        with self.assertRaises(ValueError) as context:
            transform.validate_pending_records(
                [{}],
                {"t3_struct": {0}, "t3_reflect": set()},
                ["t3_struct", "t3_reflect"],
            )

        self.assertIn("Record 0", str(context.exception))


if __name__ == "__main__":
    unittest.main()
