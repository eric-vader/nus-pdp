sources := $(wildcard *.mmd)
targets := $(patsubst %.mmd,%.pdf,$(sources))

.PHONY: all
all: $(targets)

%.pdf: %.mmd
	mmdc --pdfFit --input $< -o $@