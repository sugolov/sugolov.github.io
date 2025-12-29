---
title: Making a blog with Pandoc
author: anton
date: 2025-12-28
---



A blog is a great way to communicate, though it can only be good if it has posts. With my daily tools I find it difficult to effectively post with minimal changes to my workflow. I mostly use Obsidian for quick Markdown notes (on papers, TODOs, API tokens, social insurance number, etc.) and other tools for more involved LaTeX writeups. An easy way to make these writeups postable is to simply use `pandoc` to convert to HTML. For those unfamiliar, `pandoc` is a free and open source converter between different typesetting languages with a lot of features: this blog does a quick demo.  Spiritually, I somewhat recreated a bare bones `org-mode` with `pandoc` and Python. This avoids the emacs learning curve, keeps the HTML in other places super customizable (I can vibecode an Ising on my homepage no problem), and does not require Ruby or GO dependencies like other static site generators. I'm a fan of this approach because of how minimal it is.

## A blog using Pandoc 

At its core `pandoc` takes a document typeset in one format and convert it to another format. As a simple example you can convert from `.md` to `.html` with

```bash
pandoc post.md -s -o post.html
```

With this and some CSS you can already map Obsidian notes to a readable webpage. You can do something very similar for `.tex` to `.html` with 

```bash
pandoc notes.tex -s -o post.html
```

Most of the below commands were easy to do with just by reading `man pandoc`

### Markdown to HTML

Below is a command for blog post creation directly from my Obsidian vault.

```bash
pandoc ~/Documents/obsidian/blog/new.md -s \   # target md file, standalone
	--mathjax \                          # MathJAX for LaTeX
	--css=../../../styles.css \          # relative path to css
	--toc -N \                           # table of contents, -N for Numbering
	-B template/nav.html \               # bar that is appended to top of <body>
	-o blog/posts/new/index.html         # location
```

The table of contents (`--toc`) flag with automatic numbering (`-N`) is quite nice, and I liked that you can just append a custom navigation menu (`-B`).  I just used this for a button to return to the blog index. The title, post date, author is automatically rendered if there's a `.yaml` header at the top of the file:

```python
---
title: new
author: anton
date: 2025-12-28
---
```


### LaTeX to HTML

This comand creates a webpage from a course project in my Master's

```bash
pandoc ~/msc/2024f/mat1850/project/mat1850.tex -s \ 
	--mathjax \
	--css=../../../styles.css \ 
	-B template/mathjax-config.html \
	--shift-heading-level-by=-1 \
	-o blog/posts/latex/index.html
```

The header (`-B`) now includes my custom LaTeX shortcuts so `--mathjax` can directly render from my notes. What's nice is the ability to make `\section{}` align with `<h2>` in the CSS with `--shift-heading-level-by=-1`. Straightforward, convenient, and can be as complicated as you choose to make it


### Managing an index of `.md ` posts

**Check out [sugolov/pandoc-blog](https://github.com/sugolov/pandoc-blog) to run the example below.**

The bigger question is managing an index of `.md` posts, which is typically handled by a static site generator. Say we have something a collection in `blog/md` 

```python
md
├── BERT_denoiser.md
├── new.md
├── pandoc_blog.md
└── placeholder
    └── new.md
    
2 directories, 4 files
```


#### `build.sh`

We typically want to:

1. Export all the `blog/md/*.md` files to a public directory of `html` files like `blog/posts`.
	- I typically index by date
2. Create `blog/index.html` that sees all of `blog/posts`, and other features we might want to see in `blog/`

We can do this with the below `build.sh` in the blog directory by

```bash
# Step 1: call Pandoc to convert to Markdown
for f in md/*.md; do
	date=$(head -n 6 $f | grep date | sed "s/\-//g" | grep -o "[0-9]\+")
	pandoc "$f" -s --css=../../styles.css --toc -N --toc-depth 4 -B template/nav.html --mathjax -o "posts/$date/index.html"
done

# Step 2: build the list of posts
python3 make.py
cat index_header.html index_list.html index_footer.html > index.html
```

After `chmod +x build.sh && ./build.sh` (and adding some files) our `blog` directory now looks like

```
├── build.sh
├── index_footer.html
├── index_header.html
├── index_list.html
├── index.html
├── make.py
├── md
│   ├── BERT_denoiser.md
│   ├── new.md
│   ├── pandoc_blog.md
│   └── placeholder
│       └── new.md
├── posts
│   ├── 20240101
│   │   └── index.html
│   ├── 20250114
│   │   └── index.html
│   ├── 20250115
│   │   └── index.html
│   └── 20250215
│       └── index.html
├── styles.css
└── template
    ├── macros.html
    ├── mathjax-config.html
    └── nav.html

```
#### `make.py`

The Python script just sums strings to create a list of posts in `index_list.html`

```python
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
		'file': format_date(d) + '/index.html'
	})

# Sort by date, newest first
posts.sort(key=lambda x: x['date'], reverse=True)

html = """
<body>
<h2>posts</h2>
<ul class="post-list">
"""

for p in posts:
	html += f' <li class="post-item"><span class="post-date">{p["date"]} \
		</span><a class="post-link" href="posts/{p["file"]}">{p["title"]}</a></li>\n'
html += """ </ul>
</body>
</html>"""

Path("index_list.html").write_text(html)

# Output markdown
for p in posts:
	print(f"- [{p['date']}] [{p['title']}](posts/{p['file']})")
```

A few issues with this script:

1. the `<ul>` divs are hardcoded to my `styles.css`
2. It introduces placeholders: it's a hacky way to index a post in `blog/posts` so that it doesn't have to be remade by `pandoc` in `build.sh`
	- A concrete example is a LaTeX project. We still want the post to show up in `blog/index.html`, but we don't want `build.sh` to make it each time
	- We can hack around this by creating an empty `md/placeholders/new.md` file with a matching post date as our LaTeX project. After, we can export the latex project as in the previous section to the matching `date/index.html`
3. blog posts made on the same date have a conflict


## Takeaways

It's hacky but it works. My workflow is 

```
obsidian -> copy paste -> ./build.sh -> commit and push
```

Great, but maybe the issues above need to be polished, might be an interesting project for later if I want more complicated layouts / tags / nested pages etc.


### Other tools

Other than using raw HTML, I've experimented with static site generators, emacs`org-mode`. Maybe some. Ranked in terms of experience:

1. **Hugo**. Overall, Hugo was straightforward and easy to set up when starting from a template. The one clunky thing was editing the CSS / formatting for custom pages. Making a small change to the home page requires understanding how the author set up the site structure
2. **Jekyll.** This seems like it has the same issues as Hugo except with more Ruby dependencies and sometimes heavy React features. It's fine but I like more minimal webpages
3. **Org-mode**. Generally, I think it's pretty cool. Though it has an unpleasant learning curve, and too many macros just to do 1 simple thing. There's also a funny [stack exchange post](https://stackoverflow.com/questions/384284/how-do-i-rename-an-open-file-in-emacs) about renaming a file while in the buffer that made me rethink using emacs.
	- I remember a Linux YouTuber saying that people's OS becomes a bootloader for Emacs if you use it enough. I definitely see that now
4. **Raw HTML.** Not great but you can go a long way with Claude