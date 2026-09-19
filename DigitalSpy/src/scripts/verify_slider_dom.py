"""
Test Streamlit slider DOM interaction with Playwright using keyboard ArrowRight
"""

import time
from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()
    page.goto("http://localhost:8501", wait_until="networkidle")
    page.wait_for_timeout(3000)

    # Click on slider thumb or focus slider
    slider = page.query_selector('div[role="slider"]')
    if slider:
        slider.focus()
        print("Slider focused. Pressing PageDown...")
        # Press PageDown to advance slider significantly
        for _ in range(5):
            slider.press("PageDown")
            page.wait_for_timeout(500)
            
        page.wait_for_timeout(2000)
        body_text = page.inner_text("body")
        print("Current slider window text in DOM:")
        for line in body_text.splitlines():
            if "Window" in line or "State" in line or "Forecast" in line:
                print("  ", line)

    browser.close()
