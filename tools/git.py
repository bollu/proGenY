import os
import sys

def ga(files):
	os.system("git add " + files + " --ignore-errors")


def commit():
	os.system("git commit --template='.git-template'")


def push():
	os.system("git push origin master")



#ga("box2d_test.sublime-project")
#ga("box2d_test.sublime-workspace")

#ga("docs/doxyStyle/*")
#ga("docs/engine/Doxyfile")
#ga("docs/game/Doxyfile")

#ga("src/*")
#ga("lib /*")
#ga("bin/*")


#ga("makefile")
#ga("runner.py")

#ga(".tags")
#ga(".tags_sorted_by_file")

#ga("Tuprules.tup")
#ga(".tup/*")

#ga("git.py")
#ga(".gitignore")


os.system("git add -u")
commit();
#os.system("git status")
#push();
