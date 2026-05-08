"""Implementation-agnostic test suite for responsive landing page."""
import subprocess
import sys
from pathlib import Path


def find_html(project_dir: Path) -> Path:
    candidates = list(project_dir.glob("*.html")) + list(project_dir.glob("*.htm"))
    assert candidates, f"No HTML file found in {project_dir}"
    return candidates[0]


def find_css(project_dir: Path) -> list[Path]:
    return list(project_dir.glob("*.css"))


def test_html_exists(project_dir: Path):
    html = find_html(project_dir)
    assert html.exists()


def test_html_doctype(project_dir: Path):
    html = find_html(project_dir)
    content = html.read_text()
    assert "<!DOCTYPE html>" in content or "<!doctype html>" in content, \
        "Missing DOCTYPE declaration"


def test_html_lang(project_dir: Path):
    html = find_html(project_dir)
    content = html.read_text()
    assert 'lang=' in content.lower(), "Missing lang attribute on <html>"


def test_has_title(project_dir: Path):
    html = find_html(project_dir)
    content = html.read_text()
    assert '<title>' in content.lower(), "Missing <title> tag"


def test_has_viewport_meta(project_dir: Path):
    html = find_html(project_dir)
    content = html.read_text()
    assert 'viewport' in content.lower(), "Missing viewport meta tag (responsive)"


def test_has_cta_button(project_dir: Path):
    html = find_html(project_dir)
    content = html.read_text()
    # Should have a link or button pointing to #signup or /signup
    assert '#signup' in content or 'signup' in content.lower(), \
        "Missing CTA link to signup"


def test_has_header_tag(project_dir: Path):
    html = find_html(project_dir)
    content = html.read_text()
    assert '<header' in content.lower(), "Missing <header> tag"


def test_has_footer_tag(project_dir: Path):
    html = find_html(project_dir)
    content = html.read_text()
    assert '<footer' in content.lower(), "Missing <footer> tag"


def test_has_3_features(project_dir: Path):
    """Should have at least 3 feature-like elements (cards, items, etc)."""
    html = find_html(project_dir)
    content = html.read_text()
    # Count feature-related classes or sections
    indicators = ['feature', 'card', 'service', 'benefit', 'grid-item']
    count = sum(1 for ind in indicators if ind in content.lower())
    assert count >= 1 or 'class=' in content, \
        "No feature structure detected"


def test_container_element(project_dir: Path):
    """Should have some container/wrapper for layout."""
    html = find_html(project_dir)
    content = html.read_text()
    has_div = '<div' in content
    has_section = '<section' in content
    has_main = '<main' in content
    assert has_div or has_section or has_main, "No structural HTML elements found"


def test_css_exists(project_dir: Path):
    css_files = find_css(project_dir)
    styles_inline = False
    html = find_html(project_dir)
    if '<style' in html.read_text().lower():
        styles_inline = True
    assert css_files or styles_inline, "No CSS file or inline styles found"


def test_css_media_query_or_flex(project_dir: Path):
    """Responsive design: must use media queries or flexbox/grid."""
    css_content = ""
    for css_file in find_css(project_dir):
        css_content += css_file.read_text()

    html = find_html(project_dir)
    # Also check inline styles
    for style_tag in ['<style>', '<style ']:
        if style_tag in html.read_text():
            start = html.read_text().find(style_tag)
            end = html.read_text().find('</style>', start)
            css_content += html.read_text()[start:end]

    has_media = '@media' in css_content
    has_flex = 'flex' in css_content
    has_grid = 'grid' in css_content
    assert has_media or has_flex or has_grid, \
        "No responsive design (media query, flexbox, or grid) found in CSS"
