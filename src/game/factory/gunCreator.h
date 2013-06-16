#pragma once
#include "objectFactory.h"
#include "../../core/Process/viewProcess.h"
#include "../ObjProcessors/gunProcessor.h"
#include "../ObjProcessors/offsetProcessor.h"

#include "../../core/ObjProcessors/renderProcessor.h"
#include "../../core/ObjProcessors/phyProcessor.h"
#include "../../core/renderUtil.h"

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
		vector2 gunDim = vector2(1.0,3) * viewProc->getGame2RenderScale();
		sf::Shape *shape = new sf::RectangleShape(vector2::cast<sf::Vector2f>(gunDim));

			

		shape->setFillColor(sf::Color::Blue);

		Renderer shapeRenderer(shape);
		render.addRenderer(shapeRenderer);
		
		//offset-------------------------------------
		assert(this->parent != NULL);
		offset.parent = this->parent;
		offset.angleOffset = util::Angle::Deg(0);
		offset.posOffset = vector2(this->radius, 0);
		offset.centered = true;

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