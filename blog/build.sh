#!/bin/bash

# Step 1: call Pandoc to convert to Markdown
for f in md/*.md; do
  tag=$(head -n 8 "$f" | grep "blog-tag" | sed "s/blog-tag:[[:space:]]*//" | tr -d "\"'")
  if [ -z "$tag" ]; then
    echo "missing blog-tag in $f" >&2
    exit 1
  fi
  mkdir -p "posts/$tag"
  pandoc "$f" -s --css=../../../styles.css --css=../../blog.css --toc -N --toc-depth 3 -H template/meta_jepax.html -B ../template/nav.html --mathjax -o "posts/$tag/index.html"
done

# Step 2: built the list of posts
python3 make.py
cat index_header.html index_list.html index_footer.html > index.html
python3 ../scripts/build_home_blog.py
