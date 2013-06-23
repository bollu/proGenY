import os
import sys


def ga(files):
	os.system("git add " + files)


def commit():
	os.system("git commit -m " + "'" + sys.argv[1] + "'")


def push():
	os.system("git push origin master")

if len(sys.argv) < 2:
	sys.exit('Usage: git.py [commit-message]')

ga("box2d_test.sublime-project")
ga("box2d_test.sublime-workspace")

ga("docs/doxyStyle/*")
ga("docs/engine/Doxyfile")
ga("docs/game/Doxyfile")

ga("src/*")
ga("lib /*")
ga("bin/*")


ga("makefile")
ga("runner.py")

ga(".tags")
ga(".tags_sorted_by_file")

ga("Tuprules.tup")
ga(".tup/*")

ga("git.py")


commit();
push();