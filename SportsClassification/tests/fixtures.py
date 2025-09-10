import pytest
from pathlib import Path

@pytest.fixture
def test_image_file():
    return Path(__file__).parent / "fixtures" / "test_image.jpg"