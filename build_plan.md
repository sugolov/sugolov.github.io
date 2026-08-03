# Blog Build Plan

## Goal

Replace the current `blog/build.sh` + `blog/make.py` flow with one small Python build script that keeps the existing workflow:

1. Write markdown posts locally.
2. Keep post images and data files next to the markdown.
3. Run one command.
4. Get plain static HTML under `blog/posts/...` and a compact `blog/index.html`.

The output should stay minimal and understandable. The generated HTML should use the existing `styles.css` and the current visual style: narrow page width, compact lists, simple text links, and no unnecessary boxes.

## Proposed Source Layout

Use one folder per post when a post has assets:

```text
blog/md/
  jepax_v0/
    index.md
    images/
      ijepa_b.png
      ssl.png
    data/
      run_summary.csv
```

Still support the current flat-file layout for simple posts:

```text
blog/md/
  short_note.md
```

Build output:

```text
blog/posts/
  jepax/
    index.html
    images/
      ijepa_b.png
      ssl.png
    data/
      run_summary.csv
```

Markdown should reference assets relative to the markdown file:

```markdown
![](images/ijepa_b.png)
[data](data/run_summary.csv)
```

Those same relative paths should work after build because the asset directory structure is copied into the generated post directory.

## Build Behavior

Create a single script, likely `blog/build.py`, responsible for:

- Discovering posts from:
  - `blog/md/*.md`
  - `blog/md/*/index.md`
  - optionally `blog/md/placeholder/*.md` for index-only entries
- Parsing YAML frontmatter fields:
  - required: `title`, `date`, `blog-tag`
  - optional: `author`, `description`, `draft`
- Generating each post directory from `blog-tag`:
  - `blog-tag: jepax` -> `blog/posts/jepax/`
- Running `pandoc` once per non-placeholder post.
- Copying all non-markdown sibling files and directories from the post source folder into the generated post directory.
- Generating `blog/index.html` from the parsed post metadata.

For flat markdown files, asset copying is limited unless a convention is added. Prefer per-post folders for posts with images/data.

## Asset Copy Rules

For a folder post like `blog/md/jepax_v0/index.md`:

- Copy every sibling path except markdown files into `blog/posts/YYYYMMDD/`.
- Preserve directory structure exactly.
- Example:
  - `blog/md/jepax_v0/images/a.png` -> `blog/posts/jepax/images/a.png`
  - `blog/md/jepax_v0/data/x.csv` -> `blog/posts/jepax/data/x.csv`

Ignore transient files:

- `.DS_Store`
- hidden editor files
- cache directories
- generated HTML

Do not delete the whole output post directory by default. Instead, overwrite copied files and generated `index.html`. Add an explicit clean mode later if stale asset cleanup becomes annoying.

## Templates

Replace HTML concatenation with small templates:

```text
blog/templates/
  index.html
  post_before.html
  post_after.html
```

Keep templates plain. No dependency on Jinja unless the script starts needing real template logic. Simple Python string replacement is enough for now:

- `{{title}}`
- `{{date}}`
- `{{post_items}}`

The generated blog index should keep the current compact structure:

```html
<body class="blog-index">
<a href=".." class="nav-link">&larr; home</a>
<div class="blog-heading">
<h1>Blog</h1>
</div>
...
</body>
```

## Compatibility

Keep these links stable:

- `blog/`
- `blog/posts/jepax/`
- relative post asset links like `images/foo.png`

Keep `blog/index_header.html`, `blog/index_list.html`, and `blog/index_footer.html` only during migration if useful. The final state should not need them.

## Test Plan

After implementing the future build script:

- Run the build from `blog/`.
- Confirm `blog/posts/jepax/index.html` regenerates.
- Confirm all images referenced by `blog/md/jepax_v0/index.md` exist under `blog/posts/jepax/images/`.
- Confirm any `data/` directory copies with the same structure.
- Confirm `blog/index.html` contains clean post titles without YAML quote characters.
- Run `git diff --check`.
- Open `blog/index.html` and one generated post locally.

## Migration Steps

1. Move `blog/md/jepax_v0.md` to `blog/md/jepax_v0/index.md`.
2. Move the existing `blog/posts/jepax/images/` directory to `blog/md/jepax_v0/images/`.
3. Add `blog/build.py`.
4. Generate posts and index from `build.py`.
5. Confirm output matches the current public URLs.
6. Remove or retire `blog/build.sh`, `blog/make.py`, and the index fragment files only after the new script is trusted.
