#pragma once
#include "../../core/componentSys/processor/ObjectProcessor.h"
#include "../../include/SFML/Graphics.hpp"
#include "../../core/math/vector.h"
#include "../../core/Rendering/viewProcess.h"
#include "../../core/World/worldProcess.h"

struct CameraData{
	bool enabled;
	vector2 cameraCenter;

	//the absolute minimum coordinates the camera can posses
	vector2 minCoord;
	//the absolute maximum coordinates the camera can posses
	vector2 maxCoord;

	//the maximum amount the camera can be moved per frame
	vector2 maxMoveAmt; 

	//a bounding box is created from the center of the camera.
	//if the object moves out of the bounding box, then the camera
	//is shifted such that the object moves back into the bounding box.
	//these 
	float boxHalfW;
	float boxHalfH;

	
	vector2 v;
	float accumilator;


	CameraData(){};
};
class cameraProcessor : public ObjectProcessor{

public:
	cameraProcessor(processMgr &processManager, Settings &settings, EventManager &_eventManager) :
		ObjectProcessor("cameraProcessor") {
		
		worldProcess *world = processManager.getProcess<worldProcess>(Hash::getHash("worldProcess"));
		this->view =  processManager.getProcess<viewProcess>(Hash::getHash("viewProcess"));
		this->window = 
			processManager.getProcess<windowProcess>(Hash::getHash("windowProcess"))->getWindow();

		this->stepSize = world->getStepSize();
		this->maxAccumilation = world->getMaxAccumilation();

	};

private:
	float stepSize;
	float maxAccumilation;

	sf::RenderWindow *window;
	viewProcess *view;

	vector2 _limitCameraCoord(vector2 cameraCoord, CameraData *data);
	vector2 _limitMoveAmt(vector2 moveAmt, vector2 maxMoveAmt);

	vector2 _calcCameraMoveAmt(Object *obj, CameraData *data);
	void _simulateCamera(vector2 cameraMoveAmt, float dt, CameraData *data);

protected:
	void _onObjectAdd(Object *obj);
	void _Process(Object *obj, float dt);

	bool _shouldProcess(Object *obj){
		return obj->hasProperty("CameraData");
	};

};
