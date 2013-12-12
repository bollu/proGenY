#pragma once
#include "../controlFlow/Process.h"
#include "../controlFlow/processMgr.h"
#include "objectMgr.h"
//#include "objProcessors/renderProcessor.h"
//#include "objProcessors/phyProcessor.h"

#include "../Rendering/windowProcess.h"
#include "../World/worldProcess.h"

/*!Process to handle objectMgr 

Acts as a wrapper around objectMgr. this ensures that objectMgr is in sync with the other parts of
the Engine.

\sa objectMgr
*/
class objectMgrProcess : public Process{
private:
	objectMgr *objManager;


public:
	objectMgrProcess(processMgr &processManager, Settings &settings, eventMgr &eventManager) :
	 Process("objectMgrProcess"){

		this->objManager = new objectMgr();
		//this->_createObjectProcessors(processManager);
	}
	
	void preUpdate(){
		objManager->preProcess();
	};

	/*!causes the objManager to Process*/
	void Update(float dt){
		objManager->Process(dt);
	}

	void postDraw(){
		objManager->postProcess();
	}
	/*!add an objectProcessor to the objectMgr
	@param [in] processor the objectProcessor to be added
	*/
	void addObjectProcessor(objectProcessor *processor){
		this->objManager->addObjectProcessor(processor);
	}

	/*!returns the object Manager
	\return the objectMgr 
	*/
	objectMgr *getObjectMgr(){
		return this->objManager;
	}
};