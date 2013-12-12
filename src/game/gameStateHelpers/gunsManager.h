#pragma once
#include "../ObjProcessors/gunProcessor.h"
#include "../../core/Messaging/eventMgr.h"
#include "../factory/objectFactory.h"


class gunCreator;
class bulletCreator;
class objectMgr;
class viewProcess;

class gunsManager : public Observer{
public:
	gunsManager(eventMgr &eventManager, objectMgr &objectManager, viewProcess *viewProc, Object *player);
	void addGun(Object *gun, bool currentGun=false);


	void recieveEvent(const Hash *eventName, baseProperty *eventData);

private:
	viewProcess *viewProc_;

	Object *player_;
	
	std::vector<Object *>guns_;
	Object *currentGun_;
	//GunData *currentGunData_;

	objectMgr &objectManager_;

	//0 to guns.size() - 1
	int currentGunIndex_;

	//reloads the currentGun pointer and currentGunData pointer
	void _reloadGunPtrs();

	void _gotoNextGun(int skip);
	void _gotoPrevGun(int skip);

	
	void _switchGuns(Object *prevGun, Object *newGun);

	void _updateGunAngle(util::Angle &angle);
	void _fireGun();

};