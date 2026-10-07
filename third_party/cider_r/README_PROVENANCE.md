# CIDEr-R reference provenance

The vendored implementation is copied from
`gabrielsantosrv/coco-caption---My-changes`, remote HEAD
`b5f27535299eacb2bb4b599ac239841625030c8d`.

Copied files are the authors' `pycocoevalcap/ciderR` scorer and
`pycocoevalcap/tokenizer/ptbtokenizer.py`.  The PTB wrapper intentionally
requires the bundled Stanford CoreNLP 3.4.1 jar
(`stanford-corenlp-3.4.1.jar`, SHA256
`2fcb91bb7a111f93d71e264f4ee0e3afd19ba0dde6d21b38605088df9e940399`).
Missing Java assets are a hard error for a canonical
CIDEr-R run.  Ordinary CIDEr is never used as a substitute.

The scorer is not enabled by default until its tokenizer runtime is available
and its parity fixture has passed.
