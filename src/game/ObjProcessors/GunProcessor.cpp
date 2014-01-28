#include "GunProcessor.h"
#include "../../core/componentSys/ObjectMgrProcess.h"


GunProcessor::GunProcessor(processMgr &processManager, Settings &settings, EventManager &_eventManager) 
: ObjectProcessor("GunProcessor"){

	ObjectMgrProcess *objMgrProc = processManager.getProcess<ObjectMgrProcess>(
		Hash::getHash("ObjectMgrProcess"));

	this->viewProc = processManager.getProcess<viewProcess>(Hash::getHash("viewProcess"));
	this->objectManager = objMgrProc->getObjectMgr();
};

void GunProcessor::_Process(Object *obj, float dt){
	
	GunData *gunData = obj->getPrimitive<GunData>(Hash::getHash("GunData"));

	vector2 *spawnPos = obj->getMessage<vector2>(Hash::getHash("setBulletSpawnPos"));
	if(spawnPos != NULL){
		gunData->bulletPos = *spawnPos;
	}

	util::Angle *facing = obj->getMessage<util::Angle>(Hash::getHash("setGunFacing"));
	if(facing != NULL){
		gunData->facing = *facing;
	}

	gunData->_Tick(dt);
	if(obj->getMessage(Hash::getHash("fireGun")) != NULL && gunData->_canFire()){
		this->_fireShot(gunData, gunData->bulletPos);
		gunData->_Cooldown();
	}
};	


void GunProcessor::_onObjectDeactivate(Object *obj){
	
};

#include "../factory/objectFactories.h"
void GunProcessor::_fireShot(GunData *gunData, vector2 pos){
	ObjectFactories::BulletFactoryInfo factoryInfo;

	factoryInfo.viewProc = this->viewProc;
	factoryInfo.pos = pos;
	factoryInfo.radius = gunData->bulletRadius;

	//copy bulletData from gunData onto the bullet
	factoryInfo.bulletData = gunData->bulletData;
	factoryInfo.bulletData.beginVel = gunData->facing.toVector() * gunData->bulletVel;
	factoryInfo.bulletData.angle = util::Angle(gunData->facing);

	factoryInfo.stabData = gunData->stabData;
	factoryInfo.stabData.killOnHit = true;

	Object *bullet = ObjectFactories::CreateBullet(factoryInfo);
	this->objectManager->addObject(bullet);


};


void GunData::_Tick(float dt){

	if(this->clipCooldown.onCooldown()) {
		
		//cooldown transitioned from on to off	
		if(this->clipCooldown.Tick(dt).offCooldown()) {
			this->currentClipSize = this->totalClipSize;
		}
	}

	if(this->shotCooldown.onCooldown()) {
		this->shotCooldown.Tick(dt);
	}
};


void GunData::_Cooldown(){
	this->currentClipSize--;

	if (this->currentClipSize <= 0) {
		this->clipCooldown.startCooldown();
	} else { 
		this->shotCooldown.startCooldown();
	}
};

bool GunData::_canFire(){
	return (this->clipCooldown.offCooldown() && this->shotCooldown.offCooldown());
}