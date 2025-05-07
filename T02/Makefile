DIR_NAME:=$(shell basename $(CURDIR))
TARGETS = slides notes
TIMESTAMP:=$(shell date +%d%b%Y | tr A-Z a-z)

default: $(TARGETS)

notes:%:%.md
	make -C mmd
	make -C tex
	make -C python
	pandoc -M link-citations=true --filter pandoc-crossref --filter pandoc-include --citeproc --include-in-header=../notes-header-includes.tex --highlight-style tango -t pdf -V theme:Amurmaple -V themeoptions:amurmapleblack,nomail,delaunay -V aspectratio:169 --pdf-engine=lualatex $@.md -o $(DIR_NAME)_$@.pdf
	cp $(DIR_NAME)_$@.pdf $(DIR_NAME).$$(eval grep ^title $@.md | cut -d':' -f 2 | tr -c '[:alnum:]' ' ' | xargs | tr '[:upper:]' '[:lower:]' | tr -s ' ' | tr ' ' '-')_$@_$(TIMESTAMP).pdf
	magick -density 300 $(DIR_NAME)_$@.pdf[0] -quality 99 $(DIR_NAME)_$@.png

slides:%:%.md
	make -C mmd
	make -C tex
	make -C python
	pandoc -M link-citations=true --filter pandoc-crossref --filter pandoc-include --citeproc --include-in-header=../slides-header-includes.tex --highlight-style tango -t beamer -V theme:Amurmaple -V themeoptions:amurmapleblack,nomail,delaunay -V aspectratio:169 --pdf-engine=lualatex $@.md -o $(DIR_NAME)_$@.pdf
	cp $(DIR_NAME)_$@.pdf $(DIR_NAME).$$(eval grep ^title $@.md | cut -d':' -f 2 | tr -c '[:alnum:]' ' ' | xargs | tr '[:upper:]' '[:lower:]' | tr -s ' ' | tr ' ' '-')_$@_$(TIMESTAMP).pdf
	magick -density 300 $(DIR_NAME)_$@.pdf[0] -quality 99 $(DIR_NAME)_$@.png

listen-%:%.md
	while inotifywait -e close_write $<; do make $*; done