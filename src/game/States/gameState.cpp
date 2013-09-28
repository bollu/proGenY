#pragma once
#include "gameState.h"
#include "../../core/componentSys/Object.h"
#include "../../core/Rendering/renderUtil.h"

#include "../ObjProcessors/groundMoveProcessor.h"
#include "../ObjProcessors/cameraProcessor.h"

#include "../../core/math/AABB.h"


void gameState::_Init(){

	objectMgrProcess *objMgrProc = this->processManager->getProcess<objectMgrProcess>(
		Hash::getHash("objectMgrProcess"));
	
	this->objectManager = objMgrProc->getObjectMgr();

	this->viewProc = this->processManager->getProcess<viewProcess>(
		Hash::getHash("viewProcess"));
	
	this->_initFactory();

	vector2 playerInitPos = viewProc->view2GameCoord(vector2(300, 300));
	vector2 levelDim = viewProc->view2GameCoord(vector2(5000, 5000));
	
	
	
	this->_generateBoundary(levelDim);
	//this->_generateTerrain(0, playerInitPos, levelDim);
	this->_createEnemies(levelDim);
	this->_createDummy(levelDim);
	this->_createPlayer(playerInitPos, levelDim);


//*((int *)NULL) = 1;
	
}


#include "../factory/playerCreator.h"
#include "../factory/boundaryCreator.h"
#include "../factory/dummyCreator.h"
#include "../factory/bulletCreator.h"
#include "../factory/gunCreator.h"
#include "../factory/terrainCreator.h"
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

	this->objFactory.attachObjectCreator(Hash::getHash("gun"),
		new gunCreator(this->viewProc));

	this->objFactory.attachObjectCreator(Hash::getHash("terrain"),
		new terrainCreator(this->viewProc));

	this->objFactory.attachObjectCreator(Hash::getHash("pickup"),
		new pickupCreator(this->viewProc));

	this->objFactory.attachObjectCreator(Hash::getHash("enemy"),
		new enemyCreator(this->viewProc));
};


void gameState::_generateTerrain(unsigned long long seed, 
	vector2 playerInitPos, vector2 levelDim){

	float render2GameCoord =  viewProc->getRender2GameScale();
	terrainCreator *creator = objFactory.getCreator<terrainCreator>(
		Hash::getHash("terrain"));

	vector2 blockDim =  vector2(64, 64);
	blockDim *= render2GameCoord;


	vector2 numBlocks =  vector2(levelDim.x / blockDim.x, 
		levelDim.y / blockDim.y);

		
		 
 	/*vector2 minPos   =  vector2(0, 0);
 

 	vector2 maxPos = minPos + vector2(blockDim.x * numBlocks.x, 
 					blockDim.y * numBlocks.y);
	*/
	


 	//	minPos *= render2GameCoord;
	
	// 	maxPos *= render2GameCoord;

	creator->Init(numBlocks, blockDim, 10);

	Object *terrain = creator->createObject(vector2(0,0));
	objectManager->addObject(terrain);
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

		creator->Init(1.0f);

		vector2 pos = vector2(400, 200);
		pos *= viewProc->getRender2GameScale();

		Object *dummy = creator->createObject(pos);
		objectManager->addObject(dummy);
	}

	

	{	

		pickupCreator *creator = objFactory.getCreator<pickupCreator>(
			Hash::getHash("pickup"));

		pickupData data;
		data.onPickupEvent = Hash::getHash("addGun");
		data.addCollisionType(Hash::getHash("player"));
		data.eventData = new Prop<gunDataGenerator>(
			gunDataGenerator(gunDataGenerator::Archetype::Rocket, 
				1, 10));

		creator->Init(data, 0.7);

		vector2 pos = vector2(600, 100);
		pos *= viewProc->getRender2GameScale();

		Object *obj = creator->createObject(pos);
		objectManager->addObject(obj);
	}


	{	

		pickupCreator *creator = objFactory.getCreator<pickupCreator>(
			Hash::getHash("pickup"));

		

		pickupData data;
		data.onPickupEvent = Hash::getHash("addGun");
		data.addCollisionType(Hash::getHash("player"));
		data.eventData = new Prop<gunDataGenerator>(
			gunDataGenerator(gunDataGenerator::Archetype::machineGun, 
				1, 30));
		
		creator->Init(data, 0.7);
		
		vector2 pos = vector2(1200, 100);
		pos *= viewProc->getRender2GameScale();

		Object *obj = creator->createObject(pos);
		objectManager->addObject(obj);


	}
};


#include "../bulletColliders/pushCollider.h"
#include "../bulletColliders/damageCollider.h"
Object* gameState::_createGuns(Object *player, vector2 levelDim){
	bulletCreator *_bulletCreator = objFactory.getCreator<bulletCreator>(
		Hash::getHash("bullet"));

	gunCreator *creator = objFactory.getCreator<gunCreator>(
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

	creator->Init(player, data);

	Object *gun = creator->createObject(pos);
	objectManager->addObject(gun);

	return gun;
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
