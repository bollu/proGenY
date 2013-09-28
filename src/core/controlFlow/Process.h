#pragma once
#include <string>
#include "../IO/Hash.h"
#include "eventMgr.h"
#include "../IO/Settings.h"

/*! Used to create and run core modules that the Engine needs to function

A process is a wrapper for the core functionality that the Engine uses,
such as rendering, physics, audio, etc. The Process provides a neat abstraction
to encapsulate all of these portions of the Engine in a unified way

The Process must adhere to a certain model - it must be able to Pause, to Resume,
to Update, and so on. 

\sa processMgr
*/ 
class Process{
protected:
	const Hash *nameHash;

	/*!Protected Constructor that receives the name of the Process 

	@param [in] name The name of the Process
	*/
	Process(std::string _name){
		this->nameHash = Hash::getHash(_name);
	};

public:
	virtual ~Process(){};

	/*!Used to pause the process

	Any task that is being executed by the process should be halted until Process::Resume() is called.
	Usually, this is done by not processing in the Process::Update() and Process::preUpdate()
	*/
	virtual void Pause(){};
	/*!Used to resume a paused process

	A Process that was paused can now resume the paused task.
	*/ 
	virtual void Resume(){};

	/*!Used to run Pre-Update setup

	This can be used to setup resources that are required before Updating, or to even 
	delete unused resources.
	*/ 
	virtual void preUpdate(){};

	/*!The Process is updated 

	@param: [in] dt the time _in seconds_ between the previous Update and this one
	*/ 
	virtual void Update(float dt){};

	/*!The process draws to the screen 
	*/
	virtual void Draw(){};
	
	/*!Any cleaning up / extra rendering is done by the Process
	*/
	virtual void postDraw(){};


	//virtual void Startup() = 0;
	//
	/*!The Process shuts down, saves data and deletes all memory used by it
	*/ 
	virtual void Shutdown(){};

	/*!returns the Hash of the name
	\return the const Hash * of the Process name
	*/
	const Hash *getNameHash(){
		return this->nameHash;
	}

};
