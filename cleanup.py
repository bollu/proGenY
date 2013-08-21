import os.path
import sys
import subprocess

if len(sys.argv) < 3 :
	print("usage: cleanup.py [uncrustify config file path] [input file path]")	
	sys.exit()

if(not os.path.isfile(sys.argv[1]) ):
	print("provide a proper config file path")	
	sys.exit()

if(not os.path.isfile(sys.argv[2]) ):
	print("provide a proper input file path")	
	sys.exit()

beautifierPath = os.path.abspath(sys.argv[1])
filePath = os.path.abspath(sys.argv[2])


uncrustifyCommand = ["uncrustify -c "  + beautifierPath + " -f " + filePath]
commandOutput = subprocess.check_output(uncrustifyCommand, shell=True )

#print(commandOutput)

outputFile = open(filePath,'w')
outputFile.write(commandOutput)
outputFile.close()

