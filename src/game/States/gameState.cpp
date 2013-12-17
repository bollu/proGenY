#pragma once
#include "gameState.h"
#include "../../core/componentSys/Object.h"
#include "../../core/Rendering/renderUtil.h"

#include "../ObjProcessors/GroundMoveProcessor.h"
#include "../ObjProcessors/CameraProcessor.h"

#include "../../core/math/AABB.h"


void gameState::_Init(){

	ObjectMgrProcess *objMgrProc = this->processManager->getProcess<ObjectMgrProcess>(
		Hash::getHash("ObjectMgrProcess"));
	
	this->objectManager = objMgrProc->getObjectMgr();

	this->viewProc = this->processManager->getProcess<viewProcess>(
		Hash::getHash("viewProcess"));
	
	this->_initFactory();

	vector2 levelDim, playerInitPos;
	this->_generateTerrain(0, playerInitPos, levelDim);
	
	//this->_generateBoundary(levelDim);	
	this->_createEnemies(levelDim);
	this->_createDummy(levelDim);
	this->_createPlayer(playerInitPos, levelDim);
}


#include "../factory/playerCreator.h"
#include "../factory/boundaryCreator.h"
#include "../factory/dummyCreator.h"
#include "../factory/bulletCreator.h"

#include "../factory/pickupCreator.h"
#include "../factory/enemyCreator.h"

void gameState::_initFactory(){

	
	this->objFactory.attachObjectCreator(Hash::getHash("dummy"),
		new dummyCreator(this->viewProc));

	this->objFactory.attachObjectCreator(Hash::getHash("player"),
		new playerCreator(this->viewProc));

	this->objFactory.attachObjectCreator(Hash::getHash("boundary"),
		new boundaryCreator(this->viewProc));

	this->objFactory.attachObjectCreator(Hash::getHash("bullet"),
		new bulletCreator(this->viewProc));

	this->objFactory.attachObjectCreator(Hash::getHash("pickup"),
		new pickupCreator(this->viewProc));

	this->objFactory.attachObjectCreator(Hash::getHash("enemy"),
		new enemyCreator(this->viewProc));
};

#include "../factory/objectFactories.h"
#include "../terrainGen/terrain.h"
#include "../terrainGen/terrainGenerator.h"

void gameState::_generateTerrain(unsigned long long seed, vector2& playerInitPos, vector2& levelDim) {

	vector2 blockViewDim = vector2(128, 128);
	vector2 blockGameDim = viewProc->view2GameCoord(blockViewDim);


	Terrain terrain(100, 100);
	genTerrain(terrain, 10);
	
	//level dimensions
	levelDim = vector2(terrain.getWidth() * blockViewDim.x, terrain.getMaxHeight() * blockViewDim.y);

	//player x 
	int playerX = 2;
	vector2 playerPosTerrain = getPlayerPosTerrain(terrain, playerX);
	playerInitPos = vector2(playerPosTerrain.x * blockGameDim.x, playerPosTerrain.y * blockGameDim.y);


	ObjectFactories::TerrainFactoryInfo factoryInfo(terrain);
	factoryInfo.viewProc = this->viewProc;
	factoryInfo.blockDim = blockGameDim;

	Object *terrainObj = ObjectFactories::CreateTerrain(factoryInfo);
	this->objectManager->addObject(terrainObj);


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


#include "../generators/GunDataGenerator.h"
void gameState::_createDummy(vector2 levelDim){
	{

		dummyCreator *creator = objFactory.getCreator<dummyCreator>(
			Hash::getHash("dummy"));

		creator->Init(1.0f);

		vector2 pos = vector2(400, 200);
		pos *= viewProc->getRender2GameScale();

		Object *dummy = creator->createObject(pos);
		objectManager->addObject(dummy);
	}

	

	{	

		pickupCreator *creator = objFactory.getCreator<pickupCreator>(
			Hash::getHash("pickup"));

		PickupData data;
		data.onPickupEvent = Hash::getHash("addGun");
		data.addCollisionType(Hash::getHash("player"));
		data.eventData = new Prop<GunDataGenerator>(
			GunDataGenerator(GunDataGenerator::Archetype::Rocket, 
				1, 10));

		creator->Init(data, 0.7);

		vector2 pos = vector2(600, 300);
		pos *= viewProc->getRender2GameScale();

		Object *obj = creator->createObject(pos);
		objectManager->addObject(obj);
	}


	{	

		pickupCreator *creator = objFactory.getCreator<pickupCreator>(
			Hash::getHash("pickup"));

		

		PickupData data;
		data.onPickupEvent = Hash::getHash("addGun");
		data.addCollisionType(Hash::getHash("player"));
		data.eventData = new Prop<GunDataGenerator>(
			GunDataGenerator(GunDataGenerator::Archetype::machineGun, 
				1, 30));
		
		creator->Init(data, 0.7);
		
		vector2 pos = vector2(1200, 500);
		pos *= viewProc->getRender2GameScale();

		Object *obj = creator->createObject(pos);
		objectManager->addObject(obj);


	}
};


void gameState::_generateBoundary(vector2 levelDim){

	boundaryCreator *creator = objFactory.getCreator<boundaryCreator>(
		Hash::getHash("boundary"));

	creator->Init(levelDim, 3.0f);
	Object *boundary = creator->createObject(vector2(0, -200));


	objectManager->addObject(boundary);

}

void gameState::_createEnemies(vector2 levelDim){

	

	enemyCreator *creator = objFactory.getCreator<enemyCreator>(
		Hash::getHash("enemy"));

	Object *enemy = creator->createObject(vector2(400, 400) * viewProc->getRender2GameScale());
	objectManager->addObject(enemy);

};
