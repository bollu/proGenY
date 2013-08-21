
OUTPUT = bin/a.out

all: draw-line build run

build:
	@sudo tup upd

build-console:
	@sudo tup upd

run: $(OUTPUT)
	python runner.py $(OUTPUT)

debug: $(OUTPUT)
	gdb $(OUTPUT)

clean:
	rm -rf obj/*.o
	rm -rf $(OUTPUT)


draw-line:
	clear
	clear
	


