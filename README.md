# Progressive Pose-Guided 4D Animal Reconstruction

Static academic project page prepared for GitHub Pages.

## Structure

```text
index.html
scripts/   Video splitting tool for the local PPT exports
static/
  css/       Template and project styles
  images/    Teaser and pipeline figures
  js/        Local page scripts and icons
  videos/    Individual examples for the four video sections
```

## Public Assets

| File | Content |
| --- | --- |
| `static/images/teaser.jpg` | Paper teaser |
| `static/images/pipeline.jpg` | Method pipeline |
| `static/videos/results-gallery/*.mp4` | Seven Results Gallery examples, PPT slides 2-8 |
| `static/videos/comparison/*.mp4` | Six comparisons, PPT slides 2-7 |
| `static/videos/application/*.mp4` | Two application examples, PPT slides 10-11 |
| `static/videos/limitations/*.mp4` | Three limitation examples, PPT slides 13-15 |

The four combined MP4 files are retained locally as the source exports. To
regenerate the individual clips, run `scripts/split_gallery_videos.py` with an
`--ffmpeg` path. The script reads slide timing from the two PPTX files in the
parent directory (or `--pptx-dir`) and omits the section title slides. The
combined MP4 files are inputs to this script and are not needed by the page.

## GitHub Pages

Publish this directory at the repository root and enable GitHub Pages from the
chosen branch. `.nojekyll` keeps the static asset directories unchanged.

The page is based on the
[Academic Project Page Template](https://github.com/eliahuhorwitz/Academic-project-page-template).

The Paper and Supplementary buttons are disabled placeholders. The arXiv and
Code buttons link to the public paper page and repository.
