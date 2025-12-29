#!/usr/bin/env python3
import os
import re
from pathlib import Path


posts = []
md_dir = Path("md")
placeholder_dir = Path("md/placeholder")

def format_date(date):
    out, L = "", date.split("-")
    for s in L: out = out + str(s)
    return out


md_files = list(md_dir.glob("*.md")) + list(placeholder_dir.glob("*.md"))

for f in md_files:
    content = f.read_text()
    
    # Extract YAML frontmatter
    match = re.match(r'^---\s*\n(.*?)\n---', content, re.DOTALL)
    if not match:
        continue
    
    frontmatter = match.group(1)
    title = re.search(r'^title:\s*(.+)$', frontmatter, re.MULTILINE)
    date = re.search(r'^date:\s*(.+)$', frontmatter, re.MULTILINE)
    
    if title and date:
        t = title.group(1).strip()
        d = date.group(1).strip()
        posts.append({
            'title': t,
            'date': d,
            # 'file': f.stem + '/index.html'
            'file': format_date(d)
        })

# Sort by date, newest first
posts.sort(key=lambda x: x['date'], reverse=True)

# Output markdown
for p in posts:
    print(f"- [{p['date']}] [{p['title']}](posts/{p['file']})")

html = """
<body>
  <h2>posts</h2>
  <ul class="post-list">
"""
for p in posts:
    html += f'    <li class="post-item"><span class="date">{p["date"]}</span><a class="post-link" href="posts/{p["file"]}">{p["title"]}</a></li>\n'

html += """  </ul>
</body>
</html>"""

Path("index_list.html").write_text(html)