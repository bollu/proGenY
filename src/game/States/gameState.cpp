#pragma once
#include "gameState.h"
#include "../../core/Object.h"
#include "../../core/renderUtil.h"

#include "../ObjProcessors/groundMoveProcessor.h"
#include "../ObjProcessors/cameraProcessor.h"



void gameState::_Init(){
	Prop<int> p(10);
	int *i = (int *)p;

	objectMgrProcess *objMgrProc = this->processManager->getProcess<objectMgrProcess>(
		Hash::getHash("objectMgrProcess"));
	
	this->objectManager = objMgrProc->getObjectMgr();

	this->viewProc = this->processManager->getProcess<viewProcess>(
		Hash::getHash("viewProcess"));
	
	this->_initFactory();

	vector2 playerInitPos = viewProc->view2GameCoord(vector2(300, 300));
	vector2 levelDim = viewProc->view2GameCoord(vector2(3000, 2000));
	
	
	
	this->_generateBoundary(levelDim);
	this->_generateTerrain(0, playerInitPos, levelDim);
	this->_createEnemies(levelDim);
	this->_createDummy(levelDim);
	this->_createPlayer(playerInitPos, levelDim);
	
}


#include "../factory/playerCreator.h"
#include "../factory/boundaryCreator.h"
#include "../factory/dummyCreator.h"
#include "../factory/bulletCreator.h"
#include "../factory/gunCreator.h"
#include "../factory/terrainCreator.h"
#include "../factory/pickupCreator.h"
#include "../factory/bladeCreator.h"

void gameState::_initFactory(){

	
	this->objFactory.attachObjectCreator(Hash::getHash("dummy"),
				new dummyCreator(this->viewProc));

	this->objFactory.attachObjectCreator(Hash::getHash("player"),
				new playerCreator(this->viewProc));

	this->objFactory.attachObjectCreator(Hash::getHash("boundary"),
				new boundaryCreator(this->viewProc));

	this->objFactory.attachObjectCreator(Hash::getHash("bullet"),
				new bulletCreator(this->viewProc));

	this->objFactory.attachObjectCreator(Hash::getHash("gun"),
					new gunCreator(this->viewProc));
	this->objFactory.attachObjectCreator(Hash::getHash("terrain"),
					new terrainCreator(this->viewProc));
	this->objFactory.attachObjectCreator(Hash::getHash("pickup"),
					new pickupCreator(this->viewProc));
};


void gameState::_generateTerrain(unsigned long long seed, 
	vector2 playerInitPos, vector2 levelDim){

	terrainCreator *creator = objFactory.getCreator<terrainCreator>(
		Hash::getHash("terrain"));

 	vector2 blockDim =  vector2(64, 64);
 	vector2 terrainDim =  vector2(levelDim.x / blockDim.x, 
 		levelDim.y / blockDim.y); 
 	vector2 minPos   =  vector2(0, 0);
 

 	vector2 maxPos = minPos + vector2(blockDim.x * terrainDim.x, 
 					blockDim.y * terrainDim.y);

 	float render2GameCoord =  viewProc->getRender2GameScale();



 	minPos *= render2GameCoord;
 	blockDim *= render2GameCoord;
 	maxPos *= render2GameCoord;

	creator->setBounds(minPos, maxPos, blockDim);
	creator->reserveRectSpace(playerInitPos, 
		vector2(256, 256) * render2GameCoord);


	
	Object *terrainObj = creator->createObject();


};



void gameState::_createPlayer(vector2 playerInitPos, vector2 levelDim){

	playerCreator *creator = objFactory.getCreator<playerCreator>(
		Hash::getHash("player"));

	//players handlers--------------------------------------------------
	playerHandlerData playerData;
	playerData.left = sf::Keyboard::Key::A;
	playerData.right = sf::Keyboard::Key::D;
	playerData.up = sf::Keyboard::Key::W;
	playerData.fireGun = sf::Keyboard::Key::S;



	this->_playerController = new playerController(this->eventManager, this->objectManager,
						&this->objFactory, this->viewProc);

	this->_playerController->createPlayer(levelDim, playerInitPos, creator,
				playerData);

};


#include "../generators/gunDataGenerator.h"
void gameState::_createDummy(vector2 levelDim){
	{

		dummyCreator *creator = objFactory.getCreator<dummyCreator>(
			Hash::getHash("dummy"));

		
		creator->setRadius(1.0f);

		vector2 randPos = vector2(400, 200);
		
		randPos *= viewProc->getRender2GameScale();
		Object *dummy = creator->createObject(randPos);


		objectManager->addObject(dummy);
	}

	

	{	
	
		pickupCreator *creator = objFactory.getCreator<pickupCreator>(
			Hash::getHash("pickup"));

		creator->setCollisionRadius(1.0f);

		pickupData data;
		data.onPickupEvent = Hash::getHash("addGun");
		data.addCollisionType(Hash::getHash("player"));
		data.eventData = new Prop<gunDataGenerator>(
			gunDataGenerator(gunDataGenerator::Archetype::Rocket, 
				1, 10));
		
		
	
		creator->setPickupData(data);


		vector2 pos = vector2(600, 100);
		pos *= viewProc->getRender2GameScale();


		Object *obj = creator->createObject(pos);
		objectManager->addObject(obj);


	}


	{	
	
		pickupCreator *creator = objFactory.getCreator<pickupCreator>(
			Hash::getHash("pickup"));

		creator->setCollisionRadius(1.0f);

		pickupData data;
		data.onPickupEvent = Hash::getHash("addGun");
		data.addCollisionType(Hash::getHash("player"));
		data.eventData = new Prop<gunDataGenerator>(
			gunDataGenerator(gunDataGenerator::Archetype::machineGun, 
				1, 10));
		
		
	
		creator->setPickupData(data);


		vector2 pos = vector2(1200, 100);
		pos *= viewProc->getRender2GameScale();


		Object *obj = creator->createObject(pos);
		objectManager->addObject(obj);


	}
};


void gameState::_generateBoundary(vector2 levelDim){

	boundaryCreator *creator = objFactory.getCreator<boundaryCreator>(
		Hash::getHash("boundary"));

	creator->setBoundaryThickness(3.0f);
	creator->setDimensions(levelDim);

	Object *boundary = creator->createObject(vector2(0, -200));


	objectManager->addObject(boundary);

}

void gameState::_createEnemies(vector2 levelDim){

	/*

	bulletCreator *creator = (bulletCreator *)objFactory.getCreator(
		Hash::getHash("bullet"));


	bulletPropdata;
	data.addEnemyCollision(Hash::getHash("dummy"));


	creator->setBulletData(data);
	creator->setCollisionRadius(0.8f);

	vector2 pos = vector2(400, 400);
	pos *= viewProc->getRender2GameScale();

	Object *obj = creator->createObject(pos);
	

	objectManager->addObject(obj);
	
	*/
};
