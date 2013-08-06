#pragma once
#include "../Hash.h"
#include "../../util/logObject.h"


class Process;

/*!Handles the execution of Process

The processMgr manages the life cycle of all Processes.
It acts as a centralized hub where all Processes are
managed together.

\sa Process
*/
class processMgr{
private:
	std::map<const Hash*, Process *>processes;

	Process *_getProcess(const Hash* processName);
public:
	/*!adds a Process to it's list of Processes
	@param [in] p the Process that must be added
	*/ 
	void addProcess(Process *p);

	/*!preUpdates all Processes owned
	*/
	void preUpdate();

	/*!Updates all Processes owned
	@param [in] dt the time in _seconds_ between the previous Update and this one
	*/  
	void Update(float dt);
	/*!Draws all Processes owned
	*/
	void Draw();
	void postDraw();

	/*!Shuts down all owned Processes
	*/
	void Shutdown();
	
	/*!Used to pause a Process execution
	@param [in] processName Hash of the name of the process to be paused
	*/
	void PauseProcess(const Hash* processName);
	/*!Used to resume a Process execution
	@param [in] processName Hash of the name of the process to be resumed
	*/
	void ResumePorcess(const Hash* processName);

	/*! Used to retrieve a Process
	@param [in] processName the name of the process to be returned
	@param [in] processType the derived class type of Process

	\return returns the corresponding derived Process, or NULL if no process was found of the
			given name
	*/
	template <typename processType>
	processType *getProcess(const Hash* processName){
		processType *proc = dynamic_cast< processType * >(this->_getProcess(processName));

		if(proc == NULL){
			util::errorLog<<"trying to get a process by the wrong type. \nProcessName: "<<processName;
			return NULL;
		}

		return proc;
	};
};