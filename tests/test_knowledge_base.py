import unittest
from src.knowledge_base import find_faults, load_faults, tokenize


class KnowledgeBaseTests(unittest.TestCase):
    def test_json_is_valid(self):
        self.assertGreater(len(load_faults()), 0)

    def test_compatible_search(self):
        results = find_faults("Pompa pracuje ale woda nie leci")
        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "water_no_flow_001")

    def test_no_fake_model_confirmation(self):
        result = find_faults("nie leci woda", brand="DeLonghi", model="ETAM29.660")[0]
        self.assertFalse(result["model_compatibility_confirmed"])

    def test_brand_restriction(self):
        record = load_faults()[0]
        record["applicability"] = {"brands": ["Jura"], "models": []}
        self.assertEqual(find_faults("nie leci woda", brand="Melitta", faults=[record]), [])

    def test_empty_search(self):
        self.assertEqual(find_faults("zzxxzz nothing relevant"), [])

    def test_accent_normalization(self):
        self.assertIn("mlynek", tokenize("Młynek"))


if __name__ == "__main__":
    unittest.main()
