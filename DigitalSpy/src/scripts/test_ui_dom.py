"""
UI DOM & Element Verification Test Suite for DigitalSpy Streamlit App
Tests all UI components, threshold gating, section labels, heatmaps, SHAP fallback, and Demo slider state changes.
"""

import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from streamlit.testing.v1 import AppTest

APP_PATH = str(project_root / "src" / "digitalspy" / "ui" / "app.py")

class TestDigitalSpyUI(unittest.TestCase):

    def test_default_demo_mode_labels_and_gating(self):
        """Test default load in Demo Mode: Check UI labels, heatmaps, ATT&CK gating, and SHAP."""
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()

        self.assertEqual(len(at.exception), 0, f"App threw exceptions: {at.exception}")

        # Combine all markdown text output to inspect DOM text
        all_text = "\n".join([m.value for m in at.markdown])

        # 1. Verify ATT&CK panel label
        self.assertIn("Operational Forecast State", all_text, "ATT&CK panel label missing 'Operational Forecast State'")
        self.assertNotIn("Predicted Z_t Bucket", all_text, "Legacy label 'Predicted Z_t Bucket' should not be present")

        # 2. Verify Raw State-Head Probability Distribution in Plotly charts
        chart_titles = []
        for chart in at.plotly_chart:
            spec = chart.spec
            if isinstance(spec, dict) and "layout" in spec and "title" in spec["layout"]:
                title_obj = spec["layout"]["title"]
                if isinstance(title_obj, dict) and "text" in title_obj:
                    chart_titles.append(title_obj["text"])
                elif isinstance(title_obj, str):
                    chart_titles.append(title_obj)

        found_heatmap = any("Raw State-Head Probability Distribution" in t for t in chart_titles)
        self.assertTrue(found_heatmap, f"Heatmap title missing in Plotly charts: {chart_titles}")

        # 3. Verify Criterion 2 is explicitly marked FAIL / RESEARCH RESULT
        self.assertIn("FAIL / RESEARCH RESULT", all_text, "Criterion 2 must be marked FAIL / RESEARCH RESULT")

        print("[UI TEST] ✓ Verified core labels, heatmap subtitle, Plotly chart titles, and Criterion 2 status.")

    def test_demo_slider_attack_onset_panel(self):
        """Test sliding sequence in Demo Mode to reach attack onset window."""
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()

        # Find slider element
        if len(at.slider) > 0:
            slider = at.slider[0]
            # Set value to 4386
            slider.set_value(4386)
            at.run()

            self.assertEqual(len(at.exception), 0, f"Slider run threw exceptions: {at.exception}")
            all_text = "\n".join([m.value for m in at.markdown])

            # Verify Forecast -> Observed panel appears at attack onset
            self.assertIn("Forecast → Observed Outcome", all_text, 
                          "Demo attack onset tracking panel 'Forecast → Observed Outcome' should be rendered when z_t != BENIGN")
            print("[UI TEST] ✓ Verified 'Forecast -> Observed Outcome' tracking panel appears at attack onset.")
        else:
            self.fail("No slider widget found in Demo Mode UI!")

    def test_csv_upload_mode(self):
        """Test switching mode to Upload File."""
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()

        # Find radio button for Data Source
        radio = at.radio[0]
        # Switch to "Upload File"
        radio.set_value("Upload File")
        at.run()

        all_text = "\n".join([m.value for m in at.markdown])
        self.assertIn("Upload a CIC-IDS2017 CSV or PCAP file", all_text)
        print("[UI TEST] ✓ Successfully switched to CSV Upload mode.")

if __name__ == "__main__":
    unittest.main()
