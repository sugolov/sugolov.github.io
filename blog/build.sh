#!/bin/bash

# Step 1: call Pandoc to convert to Markdown
for f in md/*.md; do
  date=$(head -n 6 $f | grep date | sed "s/\-//g" | grep -o "[0-9]\+")
  pandoc "$f" -s --css=../../../styles.css --css=../../blog.css --toc -N --toc-depth 4 -H template/meta_jepax.html -B ../template/nav.html --mathjax -o "posts/$date/index.html"
done

# Step 2: built the list of posts
python3 make.py
cat index_header.html index_list.html index_footer.html > index.html