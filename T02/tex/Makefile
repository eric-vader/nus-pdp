sources := $(wildcard *.tex)
targets := $(patsubst %.tex,%.pdf,$(sources))

.PHONY: all
all: $(targets)

%.pdf: %.tex
	pdflatex $<
	# pdftoppm -r 100 -png $@ > $@.png 