import tempfile
import unittest
from pathlib import Path
from src.review_store import (
    change_status, get_public_status, get_admin_case, list_cases, log_search,
    list_events, list_outbox, submit_case
)

class ReviewStoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / 'db.sqlite3'

    def tearDown(self):
        self.tmp.cleanup()

    def submit(self, **overrides):
        data = dict(brand='Jura', model='E8', symptoms='Wycieka przy zaworze podczas kawy',
                    reported_cause='Możliwa uszczelka', cause_certainty='suspected',
                    db_path=self.path)
        data.update(overrides)
        return submit_case(**data)

    def test_new_case_stays_pending_and_not_in_trusted_kb(self):
        id_, created = self.submit()
        self.assertTrue(created)
        self.assertEqual('pending', get_public_status(id_, self.path)['status'])
        self.assertNotIn('symptoms', get_public_status(id_, self.path))

    def test_duplicate_returns_same_id(self):
        a, fresh = self.submit()
        b, fresh2 = self.submit()
        self.assertEqual(a, b)
        self.assertTrue(fresh)
        self.assertFalse(fresh2)
        self.assertEqual(1, len(list_cases(db_path=self.path)))

    def test_email_requires_consent(self):
        with self.assertRaises(ValueError):
            self.submit(email='abc@example.com')
        with self.assertRaises(ValueError):
            self.submit(email='fake', consent_to_email=True)

    def test_not_emailed_without_consent(self):
        id_, _ = self.submit()
        change_status(case_id=id_, new_status='approved_for_kb', db_path=self.path)
        self.assertEqual([], list_outbox(self.path))

    def test_email_prepared_only_after_review_and_never_sent(self):
        id_, _ = self.submit(email='tech@example.com', consent_to_email=True)
        self.assertEqual([], list_outbox(self.path))
        change_status(case_id=id_, new_status='in_review', db_path=self.path)
        self.assertEqual([], list_outbox(self.path))
        change_status(case_id=id_, new_status='needs_info', reviewer_note='Podaj serię urządzenia', db_path=self.path)
        prepared = list_outbox(self.path)
        self.assertEqual(1, len(prepared))
        self.assertEqual('queued', prepared[0]['state'])

    def test_no_duplicate_emails_for_repeat_status(self):
        id_, _ = self.submit(email='tech@example.com', consent_to_email=True)
        change_status(case_id=id_, new_status='closed', db_path=self.path)
        change_status(case_id=id_, new_status='closed', db_path=self.path)
        self.assertEqual(1, len(list_outbox(self.path)))

    def test_search_logs_do_not_save_raw_query(self):
        log_search(found_count=0, brand='Saeco', model='GranAroma', db_path=self.path)
        logs = list_events(db_path=self.path)
        self.assertEqual(1, len(logs))
        self.assertIn('model_provided', logs[0]['metadata_json'])
        self.assertNotIn('GranAroma', logs[0]['metadata_json'])

    def test_validation(self):
        with self.assertRaises(ValueError):
            self.submit(symptoms='')
        with self.assertRaises(ValueError):
            self.submit(symptoms='A'*1201)
        with self.assertRaises(ValueError):
            self.submit(cause_certainty='sure')

    def test_public_status_never_exposes_email(self):
        id_,_ = self.submit(email='tech@example.com', consent_to_email=True)
        self.assertNotIn('email', str(get_public_status(id_,self.path)))
        self.assertEqual('tech@example.com', get_admin_case(id_,self.path)['email'])

    def test_unknown_id_has_no_status(self):
        self.assertIsNone(get_public_status('not-exists', self.path))

if __name__ == '__main__':
    unittest.main()
