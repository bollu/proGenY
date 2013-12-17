#pragma once
#include "../controlFlow/Process.h"
#include "../controlFlow/processMgr.h"
#include "ObjectMgr.h"
//#include "objProcessors/RenderProcessor.h"
//#include "objProcessors/PhyProcessor.h"

#include "../Rendering/windowProcess.h"
#include "../World/worldProcess.h"

/*!Process to handle ObjectMgr 

Acts as a wrapper around ObjectMgr. this ensures that ObjectMgr is in sync with the other parts of
the Engine.

\sa ObjectMgr
*/
class ObjectMgrProcess : public Process{
private:
	ObjectMgr *objManager;


public:
	ObjectMgrProcess(processMgr &processManager, Settings &settings, eventMgr &eventManager) :
	 Process("ObjectMgrProcess"){

		this->objManager = new ObjectMgr();
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
	/*!add an ObjectProcessor to the ObjectMgr
	@param [in] processor the ObjectProcessor to be added
	*/
	void addObjectProcessor(ObjectProcessor *processor){
		this->objManager->addObjectProcessor(processor);
	}

	/*!returns the object Manager
	\return the ObjectMgr 
	*/
	ObjectMgr *getObjectMgr(){
		return this->objManager;
	}
};
