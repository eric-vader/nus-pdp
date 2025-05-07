sources := $(wildcard *.py)
targets := $(patsubst %.py,%,$(sources))

.PHONY: all
all: $(targets)

%: %.py
	python $<