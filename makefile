
OUTPUT = bin/a.out

build:
	sudo tup upd

run: $(OUTPUT)
	python runner.py $(OUTPUT)

clean:
	rm -rf obj/*.o
	rm -rf $(OUTPUT)

monitor:
	sudo tup monitor -b

docs: $(OUTPUT)



