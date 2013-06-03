#pragma once
#include "../../core/State/State.h"
#include "../../core/State/dummyStateSaveLoader.h"
#include "../../core/objectMgr.h"
#include "../../core/Process/objectMgrProcess.h"


#include "../eventHandlers/WSADHandler.h"
class gameState : public State{
public:
	gameState() : State("gameState"){};
	
	void Update(float dt){
		this->playerMoveHandler->Update();
	};
	
	void Draw(){};

	stateSaveLoader *createSaveLoader(){
		return new dummyStateSaveLoader();
	}
protected:
	void _Init(){
		objectMgrProcess *objMgrProc = this->processManager->getProcess<objectMgrProcess>(Hash::getHash("objectMgrProcess"));
		this->objectManager = objMgrProc->getObjectMgr();

		this->viewProc = this->processManager->getProcess<viewProcess>(Hash::getHash("viewProcess"));

		vector2 playerInitPos = viewProc->render2GameCoord(vector2(300, 300));
		vector2 levelDim = viewProc->render2GameCoord(vector2(1280 * 2 , 720 * 2));
		this->_generateBoundary(levelDim);
		this->_generateTerrain(0, playerInitPos);
		this->_createPlayer(playerInitPos);
		this->_createEnemies();

		for(int i = 0; i < 10; i++){
			this->_createDummy();
		}

		util::msgLog("inited");
		

	}

	void _generateBoundary(vector2 levelDim);
	void _generateTerrain(unsigned long long seed, vector2 playerInitPos);
	void _createPlayer(vector2 playerInitPos);
	void _createEnemies();
	void _createDummy();

	objectMgr *objectManager;
	viewProcess *viewProc;

	WSADHandler *playerMoveHandler;
};