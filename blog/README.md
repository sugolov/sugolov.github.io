# blog

Directory structure should be something like
```
blog
├── build.sh
├── index_footer.html
├── index_header.html
├── index_list.html
├── index.html
├── make.py
├── md
├── posts
└── styles.css
```

Update the blog withp

```
cp ~/Documents/obsidian/blog/doc.md md/ && ./build.sh
```

## Building details
1. Copy paste new posts into `md`
2. `chmod +x build.sh && ./build.sh`
    - `md/placeholders/` can be used to index posts not directly in `md`

`make.py`
- builds `index_list.html` based on contents of `md`

`build.sh`
- converts `md` to `html` with pandoc
- creates `blog/index.html` with `cat index_header.html index_list.html index_footer.html > index.html`
