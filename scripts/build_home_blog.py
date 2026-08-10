#!/usr/bin/env python3
from html import escape
from html.parser import HTMLParser
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
INDEX_PATH = ROOT / "index.html"
BLOG_LIST_PATH = ROOT / "blog" / "index_list.html"
BLOG_FOOTER_PATH = ROOT / "blog" / "index_footer.html"

START_MARKER = "<!-- BLOG_DROPDOWN_START -->"
END_MARKER = "<!-- BLOG_DROPDOWN_END -->"


class HomeBlogParser(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        self.parts.append(f"<{tag}{self._attrs(tag, attrs)}>")

    def handle_startendtag(self, tag, attrs):
        self.parts.append(f"<{tag}{self._attrs(tag, attrs)}>")

    def handle_endtag(self, tag):
        self.parts.append(f"</{tag}>")

    def handle_data(self, data):
        self.parts.append(escape(data, quote=False))

    def _attrs(self, tag, attrs):
        rewritten = []

        for name, value in attrs:
            if value is None:
                rewritten.append((name, None))
                continue

            if tag == "a" and name == "href":
                value = rewrite_href(value)

            if tag == "div" and name == "class":
                classes = value.split()
                if "blog-section" in classes and "home-blog-section" not in classes:
                    classes.append("home-blog-section")
                value = " ".join(classes)

            rewritten.append((name, value))

        if not rewritten:
            return ""

        return "".join(
            f" {name}" if value is None else f' {name}="{escape(value, quote=True)}"'
            for name, value in rewritten
        )

    def html(self):
        return "".join(self.parts).strip()


def rewrite_href(href):
    if href.startswith(("http://", "https://", "mailto:", "tel:", "#", "/")):
        return href
    if href.startswith("../"):
        return href[3:]
    if href.startswith("./"):
        href = href[2:]
    if href.startswith("blog/"):
        return href
    return f"blog/{href}"


def build_fragment():
    blog_list = BLOG_LIST_PATH.read_text()
    blog_footer = BLOG_FOOTER_PATH.read_text()
    blog_footer = re.sub(r"\s*</body>\s*</html>\s*$", "\n", blog_footer, flags=re.I)

    parser = HomeBlogParser()
    parser.feed(blog_list + "\n" + blog_footer)
    return parser.html()


def replace_dropdown(index_html, fragment):
    pattern = re.compile(
        rf"(?P<start>[ \t]*{re.escape(START_MARKER)}\n).*?(\n[ \t]*{re.escape(END_MARKER)})",
        re.S,
    )

    if not pattern.search(index_html):
        raise RuntimeError(f"Could not find {START_MARKER} / {END_MARKER} in {INDEX_PATH}")

    indented_fragment = "\n".join(f"  {line}" if line else "" for line in fragment.splitlines())
    return pattern.sub(
        lambda match: f"{match.group('start')}{indented_fragment}{match.group(2)}",
        index_html,
        count=1,
    )


def main():
    fragment = build_fragment()
    index_html = INDEX_PATH.read_text()
    INDEX_PATH.write_text(replace_dropdown(index_html, fragment))


if __name__ == "__main__":
    main()
