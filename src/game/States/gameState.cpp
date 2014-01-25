#include "gameState.h"
#include "../../core/componentSys/Object.h"
#include "../../core/Rendering/renderUtil.h"

#include "../ObjProcessors/GroundMoveProcessor.h"
#include "../ObjProcessors/CameraProcessor.h"

#include "../../core/math/AABB.h"
#include "../factory/objectFactories.h"


void gameState::_Init(){

	ObjectMgrProcess *objMgrProc = this->processManager->getProcess<ObjectMgrProcess>(
		Hash::getHash("ObjectMgrProcess"));
	
	this->objectManager = objMgrProc->getObjectMgr();

	this->viewProc = this->processManager->getProcess<viewProcess>(
		Hash::getHash("viewProcess"));
	

	vector2 levelDim, playerInitPos;
	this->_generateTerrain(0, playerInitPos, levelDim);
	Object *player = this->_createPlayer(playerInitPos, levelDim);
	this->_createEnemies(levelDim, player);
	this->_createDummy(levelDim);
	
}


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



Object* gameState::_createPlayer(vector2 playerInitPos, vector2 levelDim){
	//TODO - load Config >_<
	playerHandlerData playerData;
	playerData.left = sf::Keyboard::Key::A;
	playerData.right = sf::Keyboard::Key::D;
	playerData.up = sf::Keyboard::Key::W;
	playerData.fireGun = sf::Keyboard::Key::S;



	this->_playerController = new playerController(eventManager, objectManager, viewProc);
	this->_playerController->createPlayer(levelDim, playerInitPos, playerData);
	
	return this->_playerController->getPlayer();
};


#include "../generators/GunDataGenerator.h"
void gameState::_createDummy(vector2 levelDim){
	ObjectFactories::PickupFactoryInfo pickupInfo;
	pickupInfo.viewProc = this->viewProc;
	pickupInfo.radius = 1.0;
	pickupInfo.pos = vector2(400, 300) * viewProc->getRender2GameScale();;

	PickupData &data = pickupInfo.pickupData;
	data.onPickupEvent = Hash::getHash("addGun");
	data.addCollisionType(Hash::getHash("player"));

	GunGenData gunGenData;
	gunGenData.type = GunType::Rocket;
	gunGenData.power = 1;
	gunGenData.seed = 30;

	data.eventData = new Prop<GunGenData>(gunGenData);

	Object *pickup = ObjectFactories::CreatePickup(pickupInfo);
	objectManager->addObject(pickup);
	/*{

		dummyCreator *creator = objFactory.getCreator<dummyCreator>(
			Hash::getHash("dummy"));

		creator->Init(1.0f);

		vector2 pos = vector2(400, 300);
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

		GunGenData gunGenData;
		gunGenData.type = GunType::Rocket;
		gunGenData.power = 1;
		gunGenData.seed = 30;

		data.eventData = new Prop<GunGenData>(gunGenData);
		
		creator->Init(data, 0.7);
		
		vector2 pos = vector2(1200, 500);
		pos *= viewProc->getRender2GameScale();

		Object *obj = creator->createObject(pos);
		objectManager->addObject(obj);


	}*/
};

void gameState::_createEnemies(vector2 levelDim, Object *player){
	ObjectFactories::EnemyFactoryInfo enemyInfo;

	enemyInfo.viewProc = this->viewProc;
	enemyInfo.pos = vector2(600, 400) * this->viewProc->getRender2GameScale();
	enemyInfo.target = player;

	Object *enemy = ObjectFactories::CreateEnemy(enemyInfo);
	objectManager->addObject(enemy);
	

	/*enemyCreator *creator = objFactory.getCreator<enemyCreator>(
		Hash::getHash("enemy"));

	Object *enemy = creator->createObject(vector2(400, 400) * viewProc->getRender2GameScale());
	objectManager->addObject(enemy);*/

};
