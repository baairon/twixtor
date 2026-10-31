"""Offline check of suite revision lookup by the numeric version the SDK headers give each revision."""
import unittest

from ae_host import Layouts


class SuiteVersions(unittest.TestCase):
    def setUp(self):
        self.lay = Layouts()

    def test_handle_suite_revision_one_is_numeric_version_two(self):
        self.assertEqual(self.lay.suite_struct("PF Handle Suite", 2), "PF_HandleSuite1")
        self.assertIsNone(self.lay.suite_struct("PF Handle Suite", 1))

    def test_app_suite_numeric_version_six_is_revision_four(self):
        self.assertEqual(self.lay.suite_struct("PF AE App Suite", 6), "PFAppSuite4")
        self.assertEqual(self.lay.suite_struct("PF AE App Suite", 7), "PFAppSuite5")

    def test_old_aegp_utility_suite_from_the_extra_layouts(self):
        self.assertEqual(self.lay.suite_struct("AEGP Utility Suite", 11), "AEGP_UtilitySuite5")
        names = [m["name"] for m in self.lay.L["AEGP_UtilitySuite5"]["members"]]
        self.assertEqual(names[8], "AEGP_RegisterWithAEGP")
        self.assertEqual(self.lay.L["AEGP_UtilitySuite5"]["members"][8]["offset"], 0x40)

    def test_unknown_suite(self):
        self.assertIsNone(self.lay.suite_struct("No Such Suite", 1))


if __name__ == "__main__":
    unittest.main()
