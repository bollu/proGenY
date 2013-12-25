
OUTPUT = bin/a.out

all: build run
	
build:
	@sudo tup upd

run: $(OUTPUT)
	./$(OUTPUT)
	#python tools/runner.py $(OUTPUT)

clean:
	rm -rf obj/*.o
	rm -rf $(OUTPUT)

monitor:
	sudo tup monitor -b

docs: $(OUTPUT)

git:
	git add .swp .tup/* .ycm_*
	git commit -m "updating dotfiles"  


