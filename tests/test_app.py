from pathlib import Path

from streamlit.testing.v1 import AppTest


APP_PATH = Path(__file__).resolve().parents[1] / "app.py"


def test_app_renders_search_prompt_without_exceptions():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)

    assert not app.exception
    assert app.title[0].value == "Ecommerce Product Discovery"
    assert any("Enter a product name" in item.value for item in app.info)


def test_reset_filters_restores_price_defaults():
    app = AppTest.from_file(str(APP_PATH)).run(timeout=30)
    app.number_input[0].set_value(10.0)
    app.number_input[1].set_value(100.0)
    app.button[0].click().run(timeout=30)

    assert app.number_input[0].value == 0.0
    assert app.number_input[1].value == 0.0
