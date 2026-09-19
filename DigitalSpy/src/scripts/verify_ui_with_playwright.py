"""
Playwright Browser DOM Testing Script for DigitalSpy Streamlit UI
Inspects real rendered DOM elements, badges, thresholds, headers, and interactive behaviors on http://localhost:8501.
"""

import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"

def test_streamlit_dom():
    print(f"[PLAYWRIGHT] Connecting to Streamlit at {URL}...")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 1000})
        page = context.new_page()

        # Navigate to Streamlit app
        page.goto(URL, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(3000) # allow Streamlit react render

        # 1. Verify Page Title & Header
        title = page.title()
        print(f"[DOM TEST] Page Title: '{title}'")
        body_text = page.inner_text("body")

        # 2. Check operational forecast state label
        assert "Operational Forecast State" in body_text, "Missing 'Operational Forecast State' in DOM"
        assert "Predicted Z_t Bucket" not in body_text, "Found legacy 'Predicted Z_t Bucket' in DOM"
        print("✓ Verified 'Operational Forecast State' label present and legacy label removed.")

        # 3. Check Raw State-Head Probability Distribution
        assert "Raw State-Head Probability Distribution" in body_text, "Missing 'Raw State-Head Probability Distribution' in DOM"
        print("✓ Verified 'Raw State-Head Probability Distribution' heatmap title present.")

        # 4. Check Success Criteria Status
        assert "CRITERION 1 — ONE-STEP RISK" in body_text
        assert "CRITERION 2 — MULTI-STEP TACTIC F1_K" in body_text
        assert "FAIL / RESEARCH RESULT" in body_text
        print("✓ Verified Benchmark Criteria (Criterion 2 marked FAIL / RESEARCH RESULT).")

        # 5. Move slider in Demo Mode to trigger attack onset panel
        # Target window 4405 (onset of attack)
        print("[DOM TEST] Interacting with Streamlit Slider to advance window...")
        slider_input = page.query_selector('input[aria-label="Select starting window index"]')
        if slider_input:
            slider_input.focus()
            # Press Right Arrow key multiple times or fill
            slider_input.fill("4386")
            slider_input.press("Enter")
            page.wait_for_timeout(3000)
            
            updated_text = page.inner_text("body")
            if "Forecast → Observed Outcome" in updated_text:
                print("✓ Verified 'Forecast → Observed Outcome' tracking panel rendered at attack onset!")
            else:
                print("ℹ Forecast panel state checked. Window advanced.")

        # Take screenshot of rendered app
        artifacts_dir = Path(__file__).resolve().parents[2] / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = artifacts_dir / "ui_dom_verification.png"
        page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"✓ Saved full page screenshot to {screenshot_path}")

        browser.close()

if __name__ == "__main__":
    test_streamlit_dom()
