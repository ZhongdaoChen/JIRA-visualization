import unittest

from app import inject_date_filter


class InjectDateFilterTest(unittest.TestCase):
    def test_empty_dates_leave_jql_unchanged(self):
        jql = 'project = "GINFOSEC" ORDER BY created DESC'

        result = inject_date_filter(jql, None, None)

        self.assertEqual(result, jql)

    def test_start_date_only_adds_lower_bound(self):
        jql = 'project = "GINFOSEC" ORDER BY created DESC'

        result = inject_date_filter(jql, "2026-02-01", None)

        self.assertEqual(
            result,
            'project = "GINFOSEC" AND (created >= "2026-02-01" OR resolutiondate >= "2026-02-01") ORDER BY created DESC',
        )

    def test_end_date_only_adds_upper_bound(self):
        jql = 'project = "GINFOSEC" ORDER BY created DESC'

        result = inject_date_filter(jql, None, "2026-06-30")

        self.assertEqual(
            result,
            'project = "GINFOSEC" AND (created <= "2026-06-30" OR resolutiondate <= "2026-06-30") ORDER BY created DESC',
        )

    def test_full_range_replaces_existing_created_date_filter(self):
        jql = 'project = "GINFOSEC" AND created >= "2025-01-01" AND created <= "2025-12-31" ORDER BY created DESC'

        result = inject_date_filter(jql, "2026-01-01", "2026-12-31")

        self.assertEqual(
            result,
            'project = "GINFOSEC" AND (created >= "2026-01-01" AND created <= "2026-12-31" OR resolutiondate >= "2026-01-01" AND resolutiondate <= "2026-12-31") ORDER BY created DESC',
        )


if __name__ == "__main__":
    unittest.main()
