#pragma once
#include "Process.h"
#include "processMgr.h"
#include "../objectMgr.h"
#include "../ObjProcessors/renderProcessor.h"
#include "../ObjProcessors/phyProcessor.h"

#include "windowProcess.h"
#include "worldProcess.h"

class objectMgrProcess : public Process{
private:
	objectMgr *objManager;

	void _createObjectProcessors(processMgr &processManager){
		windowProcess *windowProc = processManager.getProcess<windowProcess>(Hash::getHash("windowProcess"));
		worldProcess *worldProc = processManager.getProcess<worldProcess>(Hash::getHash("worldProcess"));

		objManager->addObjectProcessor( new renderProcessor(*windowProc->getWindow()));
    	objManager->addObjectProcessor( new phyProcessor(*worldProc->getWorld()));
	};

public:
	objectMgrProcess(processMgr &processManager, Settings &settings, eventMgr &eventManager) :
	 Process("objectMgrProcess"){

		this->objManager = new objectMgr();
		this->_createObjectProcessors(processManager);
	}

	void Update(float dt){
		objManager->Process();
	}

	objectMgr *getObjectMgr(){
		return this->objManager;
	}
};