import unittest
from src.knowledge_base import find_faults, load_faults


class ExtendedKnowledgeBaseTests(unittest.TestCase):
    def test_no_duplicate_ids(self):
        records = load_faults()
        self.assertEqual(len(records), len({r['id'] for r in records}))

    def test_seria_granaroma_requires_model(self):
        no_model = find_faults('wąż wypada obok przepływomierza', brand='Saeco', top_k=50)
        self.assertNotIn('saeco_granaroma_hose_solenoid_leak', [x['id'] for x in no_model])
        with_model = find_faults('wąż wypada obok przepływomierza', brand='Saeco', model='GranAroma', top_k=50)
        self.assertEqual(with_model[0]['id'], 'saeco_granaroma_hose_solenoid_leak')

    def test_saeco_not_automatically_philips(self):
        matches = find_faults('wąż wypada obok przepływomierza', brand='Philips', model='GranAroma', top_k=50)
        self.assertNotIn('saeco_granaroma_hose_solenoid_leak', [x['id'] for x in matches])

    def test_delonghi_alias(self):
        matches = find_faults('woda w pojemniku na fusy', brand="De'Longhi", top_k=50)
        self.assertIn('delonghi_grounds_infuser_leak', [x['id'] for x in matches])

    def test_generic_leak_without_brand(self):
        matches = find_faults('woda pod ekspresem')
        self.assertIn('leak_external_generic', [x['id'] for x in matches])

    def test_brand_observation_is_not_confirmed_model(self):
        matches = find_faults('woda w pojemniku na fusy', brand='DeLonghi', model='ETAM29.660')
        self.assertTrue(matches)
        self.assertFalse(matches[0]['model_compatibility_confirmed'])

    def test_possible_cause_is_not_single_certain_diagnosis(self):
        matches = find_faults('ekspres nie grzeje')
        self.assertTrue(matches)
        self.assertGreater(len(matches[0]['possible_causes']), 1)


if __name__ == '__main__':
    unittest.main()
