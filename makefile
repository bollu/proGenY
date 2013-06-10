
OUTPUT = bin/a.out

build:
	sudo tup upd

run: $(OUTPUT)
	python runner.py $(OUTPUT)

monitor:
	sudo tup monitor
