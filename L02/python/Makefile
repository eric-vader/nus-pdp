sources := $(wildcard *.py)
targets := $(patsubst %.py,%.pdf,$(sources))

.PHONY: all
all: $(targets)

%.pdf: %.py
	@echo "Running $<"
	python $<