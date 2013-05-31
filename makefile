#
# 'make depend' uses makedepend to automatically generate dependencies 
#               (dependencies are added to end of Makefile)
# 'make'        build executable file 'mycc'
# 'make clean'  removes all .o and executable files
#

# define the C compiler to use
CC = clang++

# define any compile-time flags
CFLAGS = -std=c++11

# define any directories containing header files other than /usr/include
#
INCLUDES = -I/src/include/  

# define library paths in addition to /usr/lib
#   if I wanted to include libraries not in /usr/lib I'd specify
#   their path using -Lpath, something like:
LFLAGS = -Llib/

# define any libraries to link into executable:
#   if I want to link in libraries (libx.so or libx.a) I use the -llibname 
#   option, something like (this will link in libmylib.so and libm.so:
LIBS = -lBox2D -lsfml-graphics -lsfml-window -lsfml-system

# define the C source files
SRCS = src/main.cc src/core/Hash.cpp src/util/logObject.cpp src/core/Object.cpp src/core/renderUtil.cpp \
	   src/util/strHelper.cpp src/core/Messaging/eventMgr.cpp  \
	   src/core/Process/processMgr.cpp    src/core/Process/eventProcess.cpp
	

# define the C object files 
#
# This uses Suffix Replacement within a macro:
#   $(name:string1=string2)
#         For each word in 'name' replace 'string1' with 'string2'
# Below we are replacing the suffix .c of all words in the macro SRCS
# with the .o suffix
#
OBJS = $(SRCS:.c=.o)

# define the executable file 
MAIN = box2dTest.out
#define folder where the executable file resides
DEST_FOLDER=bin/

#
# The following part of the makefile is generic; it can be used to 
# build any executable just by changing the definitions above and by
# deleting dependencies appended to the file from 'make depend'
#

.PHONY: depend clean

all: $(MAIN)
	@echo  Simple compiler named mycc has been compiled

success-build-marker: $(MAIN)
	touch $(DEST_FOLDER)success-build-marker

$(MAIN): $(OBJS) 
	$(CC) $(CFLAGS) $(INCLUDES) -o $(DEST_FOLDER)$(MAIN) $(OBJS) $(LFLAGS) $(LIBS)

# this is a suffix replacement rule for building .o's from .c's
# it uses automatic variables $<: the name of the prerequisite of
# the rule(a .c file) and $@: the name of the target of the rule (a .o file) 
# (see the gnu make manual section about automatic variables)
.c.o:
	$(CC) $(CFLAGS) $(INCLUDES) -c $<  -o $@

#call this to build. it makes sure to remove the build marker
build:
	rm -f $(DEST_FOLDER)$(MAIN)
	make

clean:
	$(RM) *.o *~ $(MAIN)


depend: $(SRCS)
	makedepend $(INCLUDES) $^

run:
	python runner.py $(DEST_FOLDER)$(MAIN)
	#$(DEST_FOLDER)$(EXE_NAME)
	#xterm -e "$(DEST_FOLDER)$(MAIN); echo 'press any key to continue..';read -n 1 c"

# DO NOT DELETE THIS LINE -- make depend needs it