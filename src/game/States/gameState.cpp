#pragma once
#include "gameState.h"
#include "../../core/Object.h"
#include "../../core/renderUtil.h"

#include "../ObjProcessors/groundMoveProcessor.h"
#include "../ObjProcessors/cameraProcessor.h"
#include "../factory/objectFactories.h"


void gameState::_Init(){


	this->objectManager = this->processManager->getProcess<objectMgr>(
		Hash::getHash("objectMgrProcess"));

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
#include "../factory/gunCreator.h"
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
	this->objFactory.attachObjectCreator(Hash::getHash("pickup"),
					new pickupCreator(this->viewProc));

	this->objFactory.attachObjectCreator(Hash::getHash("blade"),
					new bladeCreator(this->viewProc));
};


#include "../terrainGen/terrain.h"
#include "../terrainGen/terrainGenerator.h"
void gameState::_generateTerrain(unsigned long long seed, 
	vector2& playerInitPos, vector2& levelDim){
	
	vector2 blockViewDim = vector2(64, 64);
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
	objectManager->addObject(terrainObj);
}



void gameState::_createPlayer(vector2 playerInitPos, vector2 levelDim){

	playerCreator *creator = objFactory.getCreator<playerCreator>(
		Hash::getHash("player"));

	//players handlers--------------------------------------------------
	playerHandlerData playerData;
	playerData.left = sf::Keyboard::Key::A;
	playerData.right = sf::Keyboard::Key::D;
	playerData.up = sf::Keyboard::Key::W;
	playerData.fireGun = sf::Keyboard::Key::S;



	this->_playerController = new playerController(this->eventManager, this->objectManager, this->viewProc);

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

		PickupData data;
		data.onPickupEvent = Hash::getHash("addGun");
		data.addCollisionType(Hash::getHash("player"));
		data.eventData = new Prop<GunDataGenerator>(
			GunDataGenerator(GunDataGenerator::Archetype::Rocket, 
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

		PickupData data;
		data.onPickupEvent = Hash::getHash("addGun");
		data.addCollisionType(Hash::getHash("player"));
		data.eventData = new Prop<GunDataGenerator>(
			GunDataGenerator(GunDataGenerator::Archetype::machineGun, 
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
	bulletCreator *_bulletCreator = objFactory.getCreator<bulletCreator>(
		Hash::getHash("bullet"));

	gunCreator *creator = objFactory.getCreator<gunCreator>(
		Hash::getHash("gun"));

	BulletData bullet;
	bullet.addEnemyCollision(Hash::getHash("enemy"));
	bullet.addEnemyCollision(Hash::getHash("dummy"));
	bullet.addIgnoreCollision(Hash::getHash("player"));
	bullet.addIgnoreCollision(Hash::getHash("pickup"));

	bullet.addBulletCollder(new pushCollider(20.0));
	bullet.addBulletCollder(new damageCollider(1.0));

	GunData data;
	data.setClipSize(100);
	data.setClipCooldown(0);//10.0 / 1000);
	data.setShotCooldown(0);//30.0 / 1000);

	data.setBulletRadius(0.001);
	data.setBulletData(bullet);
	data.setBulletVel(10);

	vector2 pos = vector2(400, 200);
	pos *= viewProc->getRender2GameScale();

	creator->setGunData(data);
	creator->setParent(player);


	Object *gun = creator->createObject(pos);
	objectManager->addObject(gun);

	return gun;
};


void gameState::_createEnemies(vector2 levelDim){

	/*

	bulletCreator *creator = (bulletCreator *)objFactory.getCreator(
		Hash::getHash("bullet"));


	BulletData data;
	data.addEnemyCollision(Hash::getHash("dummy"));


	creator->setBulletData(data);
	creator->setCollisionRadius(0.8f);

	vector2 pos = vector2(400, 400);
	pos *= viewProc->getRender2GameScale();

	Object *obj = creator->createObject(pos);
	

	objectManager->addObject(obj);
	
	*/
};
