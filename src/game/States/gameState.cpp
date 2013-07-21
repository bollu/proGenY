#pragma once
#include "gameState.h"
#include "../../core/Object.h"
#include "../../core/renderUtil.h"

#include "../ObjProcessors/groundMoveProcessor.h"
#include "../ObjProcessors/cameraProcessor.h"



void gameState::_Init(){

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

	terrainCreator *creator = (terrainCreator*)objFactory.getCreator(
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

/*
	Object *terrainObj = creator->createObject();
	objectManager->addObject(terrainObj);
*/	

};



void gameState::_createPlayer(vector2 playerInitPos, vector2 levelDim){

	playerCreator *creator = (playerCreator*)objFactory.getCreator(
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



	Object *currentGun = this->_createGuns(this->_playerController->getPlayer(), levelDim);
	this->_playerController->addGun(currentGun, true);

	
	
};


#include "../generators/gunDataGenerator.h"
void gameState::_createDummy(vector2 levelDim){

	{

		dummyCreator *creator = (dummyCreator*)objFactory.getCreator(
			Hash::getHash("dummy"));

		
		creator->setRadius(1.0f);

		vector2 randPos = vector2(400, 200);
		
		randPos *= viewProc->getRender2GameScale();
		Object *dummy = creator->createObject(randPos);


		objectManager->addObject(dummy);
	}

	

	{	
	
		pickupCreator *creator = (pickupCreator*)objFactory.getCreator(
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
	
		pickupCreator *creator = (pickupCreator*)objFactory.getCreator(
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


#include "../bulletColliders/pushCollider.h"
#include "../bulletColliders/damageCollider.h"
Object* gameState::_createGuns(Object *player, vector2 levelDim){
	bulletCreator *_bulletCreator = (bulletCreator *)objFactory.getCreator(
		Hash::getHash("bullet"));

	gunCreator *creator = (gunCreator*)objFactory.getCreator(
		Hash::getHash("gun"));

	bulletData bullet;
	bullet.addEnemyCollision(Hash::getHash("enemy"));
	bullet.addEnemyCollision(Hash::getHash("dummy"));
	bullet.addIgnoreCollision(Hash::getHash("player"));
	bullet.addIgnoreCollision(Hash::getHash("pickup"));

	bullet.addBulletCollder(new pushCollider(2.0));
	bullet.addBulletCollder(new damageCollider(1.0));

	gunData data;
	data.setClipSize(100);
	data.setClipCooldown(100);
	data.setShotCooldown(3);
	data.setBulletRadius(0.3);
	data.setBulletCreator(_bulletCreator);
	data.setBulletData(bullet);
	data.setBulletVel(40);

	vector2 pos = vector2(400, 200);
	pos *= viewProc->getRender2GameScale();

	creator->setGunData(data);
	creator->setParent(player);


	Object *gun = creator->createObject(pos);
	objectManager->addObject(gun);

	return gun;
};


void gameState::_generateBoundary(vector2 levelDim){

	boundaryCreator *creator = (boundaryCreator*)objFactory.getCreator(
		Hash::getHash("boundary"));

	creator->setBoundaryThickness(1.0f);
	creator->setDimensions(levelDim);

	Object *boundary = creator->createObject(vector2(0, 0));


	objectManager->addObject(boundary);

}

void gameState::_createEnemies(vector2 levelDim){

	/*

	bulletCreator *creator = (bulletCreator *)objFactory.getCreator(
		Hash::getHash("bullet"));


	bulletData data;
	data.addEnemyCollision(Hash::getHash("dummy"));


	creator->setBulletData(data);
	creator->setCollisionRadius(0.8f);

	vector2 pos = vector2(400, 400);
	pos *= viewProc->getRender2GameScale();

	Object *obj = creator->createObject(pos);
	

	objectManager->addObject(obj);
	
	*/
};
