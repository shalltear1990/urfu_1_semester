from streamlit.testing.v1 import AppTest


def test_successful_streamlit_app_run():
    """Общий тест, проверяющий, запускается ли приложение вообще."""

    at = AppTest.from_file("streamlit_app.py")
    at.run()
    assert not at.exception
