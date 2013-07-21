#pragma once
#include "../ObjProcessors/gunProcessor.h"
#include "../../core/Messaging/eventMgr.h"
#include "../factory/objectFactory.h"


class gunCreator;
class bulletCreator;
class objectMgr;

class gunsManager : public Observer{
public:
	gunsManager(eventMgr &eventManager, objectFactory &factory, 
		objectMgr &objectManager, Object *player);
	void addGun(Object *gun, bool currentGun=false);


	void recieveEvent(const Hash *eventName, baseProperty *eventData);

private:
	Object *player;
	
	std::vector<Object *>guns;
	Object *currentGun;
	gunData *currentGunData;


	gunCreator *_gunCreator;
	bulletCreator *_bulletCreator;
	objectMgr &objectManager;
	//0 to guns.size() - 1
	unsigned int currentGunIndex;

	//reloads the currentGun pointer and currentGunData pointer
	void _reloadGunPtrs();

	void _gotoNextGun();
	void _gotoPrevGun();

	void _updateGunAngle(util::Angle &angle);
	void _fireGun();

};