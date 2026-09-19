"""
Complete Playwright UI DOM Verification Suite for DigitalSpy
Renders the Streamlit app in headless Chromium, performs user interactions, and validates all UI requirements.
"""

import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"

def run_dom_verification():
    results = {}
    print(f"\n==================================================")
    print(f"🚀 DIGITALSPY UI DOM VERIFICATION SUITE")
    print(f"==================================================\n")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1400, "height": 1100})
        page = context.new_page()

        print(f"[1/5] Loading Streamlit UI at {URL}...")
        page.goto(URL, wait_until="networkidle", timeout=30000)
        page.wait_for_timeout(4000)

        # ----------------------------------------------------
        # TEST 1: ATT&CK Panel Relabeling & Gating
        # ----------------------------------------------------
        body_text = page.inner_text("body")
        has_op_state = "Operational Forecast State" in body_text
        no_legacy_label = "Predicted Z_t Bucket" not in body_text
        results["ATT&CK Panel Relabeled"] = has_op_state and no_legacy_label
        print(f"  └─ Operational Forecast State Label: {'✅ PASS' if results['ATT&CK Panel Relabeled'] else '❌ FAIL'}")

        # ----------------------------------------------------
        # TEST 2: Heatmap Title & Subtitle
        # ----------------------------------------------------
        has_heatmap_title = "Raw State-Head Probability Distribution" in body_text
        has_heatmap_sub = "analytical distribution" in body_text.lower() or "project-defined network-state buckets" in body_text.lower()
        results["Heatmap Title & Subtitle"] = has_heatmap_title and has_heatmap_sub
        print(f"  └─ Heatmap Relabeling & Subtitle: {'✅ PASS' if results['Heatmap Title & Subtitle'] else '❌ FAIL'}")

        # ----------------------------------------------------
        # TEST 3: 90% Operational Threshold Chart Line
        # ----------------------------------------------------
        has_threshold_line = "90% operational threshold" in body_text
        results["90% Risk Threshold Annotation"] = has_threshold_line
        print(f"  └─ Risk Chart 90% Threshold Line: {'✅ PASS' if results['90% Risk Threshold Annotation'] else '❌ FAIL'}")

        # ----------------------------------------------------
        # TEST 4: Benchmark Criteria (Criterion 2 marked FAIL / RESEARCH RESULT)
        # ----------------------------------------------------
        has_c1 = "CRITERION 1" in body_text or "Criterion 1" in body_text
        has_c2_fail = "FAIL / RESEARCH RESULT" in body_text
        results["Criterion 2 Marked FAIL/RESEARCH RESULT"] = has_c1 and has_c2_fail
        print(f"  └─ Criterion 2 Research Status: {'✅ PASS' if results['Criterion 2 Marked FAIL/RESEARCH RESULT'] else '❌ FAIL'}")

        # ----------------------------------------------------
        # TEST 5: Interactive Slider & Attack Onset Panel
        # ----------------------------------------------------
        print("[2/5] Testing Demo Slider interaction for Forecast → Observed Tracking...")
        slider_div = page.query_selector('[data-testid="stSlider"]')
        thumb = slider_div.query_selector('[tabindex="0"]') if slider_div else None
        if thumb:
            thumb.focus()
            for _ in range(10):
                page.keyboard.press("ArrowRight")
                time.sleep(0.05)
            page.wait_for_timeout(3000)
            
            post_slider_text = page.inner_text("body")
            has_forecast_observed = "Forecast → Observed Outcome" in post_slider_text
            results["Forecast -> Observed Panel at Onset"] = has_forecast_observed
            print(f"  └─ Forecast → Observed Tracking Panel: {'✅ PASS' if has_forecast_observed else '❌ FAIL'}")
        else:
            results["Forecast -> Observed Panel at Onset"] = False
            print("  └─ Slider thumb element not found!")

        # ----------------------------------------------------
        # TEST 6: Upload Mode Switch
        # ----------------------------------------------------
        print("[3/5] Testing Sidebar Mode Switch to Upload File...")
        upload_radio = page.query_selector('text="Upload File"')
        if upload_radio:
            upload_radio.click()
            page.wait_for_timeout(2500)
            upload_text = page.inner_text("body")
            has_uploader = "Upload a CIC-IDS2017 CSV or PCAP file" in upload_text or "Drag and drop file here" in upload_text or "Browse files" in upload_text
            results["CSV Upload Mode Functional"] = has_uploader
            print(f"  └─ CSV / PCAP Upload Mode: {'✅ PASS' if has_uploader else '❌ FAIL'}")

        # Save Screenshot
        artifacts_dir = Path(__file__).resolve().parents[2] / "artifacts"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        screenshot_path = artifacts_dir / "final_ui_verification_screenshot.png"
        page.screenshot(path=str(screenshot_path), full_page=True)
        print(f"\n[4/5] Full DOM Page Screenshot saved to:")
        print(f"      {screenshot_path}")

        browser.close()

    print(f"\n==================================================")
    print(f"📊 SUMMARY REPORT")
    print(f"==================================================")
    all_pass = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        if not passed:
            all_pass = False
        print(f"{test_name:<40} : {status}")
    print(f"==================================================\n")
    
    return all_pass

if __name__ == "__main__":
    success = run_dom_verification()
    sys.exit(0 if success else 1)
