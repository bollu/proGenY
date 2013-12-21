#pragma once
#include "../controlFlow/Process.h"
#include "../controlFlow/processMgr.h"
#include "ObjectManager.h"
//#include "objProcessors/RenderProcessor.h"
//#include "objProcessors/PhyProcessor.h"

#include "../Rendering/windowProcess.h"
#include "../World/worldProcess.h"

/*!Process to handle ObjectManager 

Acts as a wrapper around ObjectManager. this ensures that ObjectManager is in sync with the other parts of
the Engine.

\sa ObjectManager
*/
class ObjectMgrProcess : public Process{
private:
	ObjectManager *objManager;


public:
	ObjectMgrProcess(processMgr &processManager, Settings &settings, EventManager &eventManager) :
	 Process("ObjectMgrProcess"){

		this->objManager = new ObjectManager();
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
	/*!add an ObjectProcessor to the ObjectManager
	@param [in] processor the ObjectProcessor to be added
	*/
	void addObjectProcessor(ObjectProcessor *processor){
		this->objManager->addObjectProcessor(processor);
	}

	/*!returns the object Manager
	\return the ObjectManager 
	*/
	ObjectManager *getObjectMgr(){
		return this->objManager;
	}
};
