#pragma once
#include "objectFactory.h"
#include "../../core/Process/viewProcess.h"
#include "../ObjProcessors/gunProcessor.h"
#include "../ObjProcessors/offsetProcessor.h"

#include "../../core/ObjProcessors/renderProcessor.h"
#include "../../core/ObjProcessors/phyProcessor.h"
#include "../../core/renderUtil.h"

#include "../defines/renderingLayers.h"


class gunCreator : public objectCreator{
private:
	viewProcess *viewProc;
	const Hash *enemyCollision;

	Object *parent;
	gunData gun;

	float radius;
public:

	gunCreator(viewProcess *_viewProc) : viewProc(_viewProc), parent(NULL), radius(0){}

	void setParent(Object *parent){
		this->parent = parent;
	}

	void setGunData(gunData gun){
		this->gun = gun;
	}

	void setOffset(float radius){
		this->radius = 0;
	}

	Object *createObject(vector2 gunPos) const{
		renderData render;
		offsetData offset;
		
		Object *gun = new Object("gun");

		vector2 *pos = gun->getProp<vector2>(Hash::getHash("position"));
		*pos = gunPos;


		//renderer------------------------------------
		vector2 gunDim = vector2(2,1) * viewProc->getGame2RenderScale();
		sf::Shape *shape = renderUtil::createRectangleShape(gunDim);
		shape->setFillColor(sf::Color::Blue);
		shape->setOutlineColor(sf::Color::White);
		shape->setOutlineThickness(-2.0);

		shapeRenderNode* renderer = new shapeRenderNode(shape, renderingLayers::aboveAction);
		render.addRenderer(renderer);
		//render.centered = true;
		
		//offset-------------------------------------
		assert(this->parent != NULL);
		offset.parent = this->parent;
		offset.offsetAngle = false;
		

		//final---------------------------------
		gun->addProp(Hash::getHash("renderData"), 
			new Prop<renderData>(render));
		gun->addProp(Hash::getHash("gunData"), 
			new Prop<gunData>(this->gun));
		gun->addProp(Hash::getHash("offsetData"), 
			new Prop<offsetData>(offset));
		
		return gun;
	};
};