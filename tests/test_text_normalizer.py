import os
import sys

import pytest

# Ensure the project root is on the import path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from text_normalizer import normalize_text

def test_normalize_removes_fillers_and_whitespace():
    result = normalize_text("Um, this   is a Test!!")
    assert result == "this is a test"

def test_normalize_maps_terms():
    text = "Setting up the API in Prod"
    result = normalize_text(text)
    assert result == "deployment the application programming interface in production"
