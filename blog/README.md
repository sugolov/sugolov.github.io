# blog

## Workflow

Add or edit posts in `blog/md/`, then rebuild from `blog/`:

```sh
cp ~/Documents/obsidian/blog/doc.md md/
./build.sh
```

That regenerates:

- `posts/<blog-tag>/index.html` for each Markdown post
- `index_list.html` from the posts in `md/`
- `index.html` from `index_header.html`, `index_list.html`, and `index_footer.html`
- the homepage Blog dropdown from `index_list.html` and `index_footer.html`

Use `md/placeholder/` for items that should appear in the index without generating a full post page.
